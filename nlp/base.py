import argparse
import evaluate
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import set_seed
from accelerate import Accelerator, DistributedType


MAX_GPU_BS = 64
EVAL_BS = 128


def get_dl(acccelerator: Accelerator, bs=16):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    datasets = ""

    def tokenize_function(examples):
        outputs = tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=None)
        return outputs

    with accelerator.main_process_first():
        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=['idx', 'sentence1', 'sentence2'],
        )


    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

    def collate_fn(examples):
        max_length = None
        if accelerator.mixed_precision == 'fp8':
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != 'no':
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None

        return tokenizer.pad(
            examples,
            padding='longest',
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors='pt'
        )

    train_dl = 
