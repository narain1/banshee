import json, os, re
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizerBase
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers.data.data_collator import DataCollatorForTokenClassification
from spacy.lang.en import English

from transformers.trainer_utils import EvalPrediction

# HF defaults
os.environ['TRANSFORMERS_CACHE'] = '/scratch/npattab1/hf_cache'
os.environ['HF_HOME'] = '/scratch/npattab1/hf_cache'
access_token = "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"

# Defaults
TRAINING_MAX_LENGTH = 3072
EVAL_MAX_LENGTH = 3072
N_SPLITS = 5
NEGATIVE_RATIO = 0.3

all_labels = [
        'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O'
    ]
id2label = {i: l for i, l in enumerate(all_labels)}
label2id = {v: k for k, v in id2label.items()}
target = [l for l in all_labels if l != "O"]

def load_data():
    with open("train.json") as f:
        data = json.load(f)

    with open("mpware_mixtral8x7b_v1.1-no-i-username.json") as f:
        extra_data = json.load(f)

    ds = DatasetDict()
    for key, data in zip(["original", "extra"], [data, extra_data]):
        ds[key] = Dataset.from_dict({
            "full_text": [x["full_text"] for x in data],
            "document": [str(x["document"]) for x in data],
            "tokens": [x["tokens"] for x in data],
            "trailing_whitespace": [x["trailing_whitespace"] for x in data],
            "provided_labels": [x["labels"] for x in data],
        })
    return ds

class TokenClassificationDataset(Dataset):
    def __init__(self, datas, tokenizer):
        self.datas = datas 
        self.tokenizer = tokenizer


class CustomTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, label2id: dict, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __call__(self, example: dict) -> dict:
        # rebuild text from tokens
        text, labels, token_map = [], [], []

        for idx, (t, l, ws) in enumerate(
            zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"])
        ):
            text.append(t)
            labels.extend([l] * len(t))
            token_map.extend([idx]*len(t))

            if ws:
                text.append(" ")
                labels.append("O")
                token_map.append(-1)

        text = "".join(text)
        labels = np.array(labels)

        # actual tokenization
        tokenized = self.tokenizer(
            "".join(text),
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length
        )

        token_labels = []

        for start_idx, end_idx in tokenized.offset_mapping:
            # CLS token
            if start_idx == 0 and end_idx == 0:
                token_labels.append(self.label2id["O"])
                continue

            # case when token starts with whitespace
            if text[start_idx].isspace():
                start_idx += 1

            token_labels.append(self.label2id[labels[start_idx]])

        length = len(tokenized.input_ids)

        return {**tokenized, "labels": token_labels, "length": length, "token_map": token_map}


def find_span(target: list[str], document: list[str]) -> list[list[int]]:
    idx = 0
    spans = []
    span = []

    for i, token in enumerate(document):
        if token != target[idx]:
            idx = 0
            span = []
            continue
        span.append(i)
        idx += 1
        if idx == len(target):
            spans.append(span)
            span = []
            idx = 0
            continue

    return spans


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1+(beta**2))*p*r / ((beta**2)*p + r + 1e-100)
        return fbeta

    def to_dict(self) -> dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}

class MetricsComputer:
    nlp = English()

    def __init__(self, eval_ds: Dataset, label2id: dict, conf_thresh: float = 0.9) -> None:
        self.ds = eval_ds.remove_columns("labels").rename_columns({"provided_labels": "labels"})
        self.gt_df = self.create_gt_df(self.ds)
        self.label2id = label2id
        self.confth = conf_thresh
        self._search_gt()

    def __call__(self, eval_preds: EvalPrediction) -> dict:
        pred_df = self.create_pred_df(eval_preds.predictions)
        return self.compute_metrics_from_df(self.gt_df, pred_df)

    def _search_gt(self) -> None:
        email_regex = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
        phone_num_regex = re.compile(r"(\(\d{3}\)\d{3}\-\d{4}\w*|\d{3}\.\d{3}\.\d{4})\s")
        self.emails = []
        self.phone_nums = []

        for _data in self.ds:
            # email
            for token_idx, token in enumerate(_data["tokens"]):
                if re.fullmatch(email_regex, token) is not None:
                    self.emails.append(
                        {"document": _data["document"], "token": token_idx, "label": "B-EMAIL", "token_str": token}
                    )
            # phone number
            matches = phone_num_regex.findall(_data["full_text"])
            if not matches:
                continue
            for match in matches:
                target = [t.text for t in self.nlp.tokenizer(match)]
                matched_spans = find_span(target, _data["tokens"])
            for matched_span in matched_spans:
                for intermediate, token_idx in enumerate(matched_span):
                    prefix = "I" if intermediate else "B"
                    self.phone_nums.append(
                        {"document": _data["document"], "token": token_idx, "label": f"{prefix}-PHONE_NUM", "token_str": _data["tokens"][token_idx]}
                    )

    @staticmethod
    def create_gt_df(ds: Dataset):
        gt = []
        for row in ds:
            for token_idx, (token, label) in enumerate(zip(row["tokens"], row["labels"])):
                if label == "O":
                    continue
                gt.append(
                    {"document": row["document"], "token": token_idx, "label": label, "token_str": token}
                )
        gt_df = pd.DataFrame(gt)
        gt_df["row_id"] = gt_df.index

        return gt_df

    def create_pred_df(self, logits: np.ndarray) -> pd.DataFrame:
        """
        Note:
            Thresholing is doen on logits instead of softmax, which could find better models on LB.
        """
        prediction = logits
        o_index = self.label2id["O"]
        preds = prediction.argmax(-1)
        preds_without_o = prediction.copy()
        preds_without_o[:,:,o_index] = 0
        preds_without_o = preds_without_o.argmax(-1)
        o_preds = prediction[:,:,o_index]
        preds_final = np.where(o_preds < self.confth, preds_without_o , preds)

        pairs = set()
        processed = []

        # Iterate over document
        for p_doc, token_map, offsets, tokens, doc in zip(
            preds_final, self.ds["token_map"], self.ds["offset_mapping"], self.ds["tokens"], self.ds["document"]
        ):
            # Iterate over sequence
            for p_token, (start_idx, end_idx) in zip(p_doc, offsets):
                label_pred = id2label[p_token]

                if start_idx + end_idx == 0:
                    # [CLS] token i.e. BOS
                    continue

                if token_map[start_idx] == -1:
                    start_idx += 1

                # ignore "\n\n"
                while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                    start_idx += 1

                if start_idx >= len(token_map):
                    break

                token_id = token_map[start_idx]
                pair = (doc, token_id)

                # ignore "O", preds, phone number and  email
                if label_pred in ("O", "B-EMAIL", "B-PHONE_NUM", "I-PHONE_NUM") or token_id == -1:
                    continue

                if pair in pairs:
                    continue

                processed.append(
                    {"document": doc, "token": token_id, "label": label_pred, "token_str": tokens[token_id]}
                )
                pairs.add(pair)

        pred_df = pd.DataFrame(processed + self.emails + self.phone_nums)
        pred_df["row_id"] = list(range(len(pred_df)))

        return pred_df

    def compute_metrics_from_df(self, gt_df, pred_df):
        """
        Compute the LB metric (lb) and other auxiliary metrics
        """

        references = {(row.document, row.token, row.label) for row in gt_df.itertuples()}
        predictions = {(row.document, row.token, row.label) for row in pred_df.itertuples()}

        score_per_type = defaultdict(PRFScore)
        references = set(references)

        for ex in predictions:
            pred_type = ex[-1] # (document, token, label)
            if pred_type != 'O':
                pred_type = pred_type[2:] # avoid B- and I- prefix

            if pred_type not in score_per_type:
                score_per_type[pred_type] = PRFScore()

            if ex in references:
                score_per_type[pred_type].tp += 1
                references.remove(ex)
            else:
                score_per_type[pred_type].fp += 1

        for doc, tok, ref_type in references:
            if ref_type != 'O':
                ref_type = ref_type[2:] # avoid B- and I- prefix

            if ref_type not in score_per_type:
                score_per_type[ref_type] = PRFScore()
            score_per_type[ref_type].fn += 1

        totals = PRFScore()

        for prf in score_per_type.values():
            totals += prf

        return {
            "precision": totals.precision,
            "recall": totals.recall,
            "f5": totals.f5,
            **{
                f"{v_k}-{k}": v_v
                for k in set([l[2:] for l in self.label2id.keys() if l!= 'O'])
                for v_k, v_v in score_per_type[k].to_dict().items()
            },
        }

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large",
                                              padding=True,
                                              truncation=True,
                                              cache_dir='/scratch/npattab1/llms/')
    model = AutoModelForTokenClassification.from_pretrained("microsoft/deberta-v3-large", 
                                                            cache_dir='/scratch/npattab1/llms/',
                                                            num_labels=len(id2label),
                                                            id2label=id2label,
                                                            label2id=label2id,
                                                            ignore_mismatched_sizes=True)

    train_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=TRAINING_MAX_LENGTH)
    eval_encoder = CustomTokenizer(tokenizer=tokenizer, label2id=label2id, max_length=EVAL_MAX_LENGTH)

    ds = load_data()
    folds = [
        (
            np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS != s]),
            np.array([i for i, d in enumerate(ds["original"]["document"]) if int(d) % N_SPLITS == s])
        )
        for s in range(N_SPLITS)
    ]
    
    negative_idxs = [i for i, labels in enumerate(ds["original"]["provided_labels"]) if not any(np.array(labels) != "O")]
    exclude_indices = negative_idxs[int(len(negative_idxs) * NEGATIVE_RATIO):]

    for fold_idx, (train_idx, eval_idx) in enumerate(folds):
        if fold_idx != 0: continue
        original_ds = ds["original"].select([i for i in train_idx if i not in exclude_indices])
        train_ds = concatenate_datasets([original_ds, ds["extra"]])
        train_ds = train_ds.map(train_encoder, num_proc=os.cpu_count())
        eval_ds = ds["original"].select(eval_idx)
        eval_ds = eval_ds.map(eval_encoder, num_proc=os.cpu_count())
        print(train_ds[0])
        train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16))

    batch = next(iter(train_dl))
    print(batch)
    print(model)