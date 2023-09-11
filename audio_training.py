import os
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
)
import torchaudio.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import librosa
import warnings
warnings.simplefilter('ignore')

def read_audio(p, target_sr = 16_0000):
    audio, sr = torch.load(p)
    resample = F.resample(audio, sr, target_sr, lowpass_filter_width=6)
    return resample


class ASRDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.df = df
        self.is_test = is_test

    def __getitem__(self, idx):
        audio = read_audio(self.df.loc[idx]['path'])
        audio = processor(
            audio,
            sampling_rate=16_000
        ).input_values[0]

        if self.is_test:
            return {'audio': audio, 'label': -1}

        else:
            with processor.as_target_processor():
                labels = processor(self.df.loc[idx]['sentence']).input_ids
            return {'audio': audio, 'label': labels}

    def __len__(self):
        return len(self.df)


def ctc_data_collator(batch):
    """
        Custom data collator function to dynamically pad the data
    """
    input_features = [{"input_values": sample["audio"]} for sample in batch]
    label_features = [{"input_ids": sample["label"]} for sample in batch]
    batch = processor.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )
    with processor.as_target_processor():
            labels_batch = processor.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )

    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
    batch["labels"] = labels
    return batch


def train_one_epoch(model, train_loader, optimizer, scheduler, device='cuda'):
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader))
    avg_loss = 0
    for data in pbar:
        data = {k: v.to(device) for k, v in data.items()}
        loss = model(**data).loss
        loss_itm = loss.item()

        avg_loss += loss_itm
        pbar.set_description(f"loss: {loss_itm:4f}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    return avg_loss/len(train_loader)


@torch.no_grad()
def valid_one_epoch(model, val_loader, device='cuda'):
    pbar = tqdm(val_loader, total=len(val_loader))
    avg_loss = 0
    for data in pbar:
        data = {k: v.to(device) for k,v in data.items()}
        loss = model(**data).loss
        loss_itm = loss.item()

        avg_loss += loss_itm
        pbar.set_description(f"val_loss: {loss_itm:4f}")

    return avg_loss / len(val_loader)


ds_root = '/scratch/npattab1/audio/data'
train_auds = '/scratch/npattab1/audio/data/train_mp3s'
df = pd.read_csv(os.path.join(ds_root, 'train.csv'))

kf = KFold(3, shuffle=True, random_state=2023)
ids = set(list(kf.split(df))[0][1])
df = df.iloc[ids, :].reset_index(drop=True)

kf = KFold(5, shuffle=True, random_state=2023)
train_ids, val_ids = list(kf.split(df))[0]

train_df = df.loc[train_ids, :].reset_index(drop=True)
val_df = df.loc[val_ids, :].reset_index(drop=True)

tokenizer = Wav2Vec2CTCTokenizer('ai4bharat/indicwav2vec_v1_bengali',
                                 unk_token='[UNK]',
                                 pad_token='[PAD]',
                                 word_delimiter_token='__')

feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False
        )
processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer
        )

model = Wav2Vec2ForCTC.from_pretrained(
            'ai4bharat/indicwav2vec_v1_bengali',
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(tokenizer),
)

model.to('cuda')
model.freeze_feature_encoder()
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=3e-4, 
    weight_decay=1e-5
)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    T_mult=2,
    eta_min=1e-6
)

# Construct training and validation dataloaders
train_ds = ASRDataset(train_df)
valid_ds = ASRDataset(val_df)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    collate_fn=ctc_data_collator,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=64,
    collate_fn=ctc_data_collator,
)


best_loss = float('inf')
for epoch in range(15):
    print(f"{'='*40} Epoch: {epoch+1} / {15} {'='*40}")
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
    valid_loss = valid_one_epoch(model, valid_loader)
    print(f"train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}")

    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), f"wav2vec2_base_bengaliAI.pt")
        print(f"Saved the best model so far with val_loss: {valid_loss:.4f}")
