from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from accelerate import Accelerator
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


