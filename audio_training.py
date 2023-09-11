import os
import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor
)
import torchaudio.functional as F

import librosa
import warnings
warnings.simplefilter('ignore')

def read_audio(p, target_sr = 16_0000):
    audio, sr = torch.load(p)
    resample = F.resample(audio, sr, target_sr, lowpass_filter_width=6)
    return resample


def construct_vocab(texts):
    all_texts = ' '.join(texts)
    vocab = list(set(all_texts))
    return vocab


