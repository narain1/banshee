import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from PIL import Image, ImageDraw
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import random


train_mask_path = '/scratch/npattab1/segmentation/data4/train_masks'
test_mask_path = '/scratch/npattab1/segmentation/data4/test_masks'
train_img_root = '/scratch/npattab1/segmentation/data4/train_data/train'
test_img_root = '/scratch/npattab1/segmentation/data4/test_data/test'

with open('ChestX_Det_train.json') as f:
    o = json.load(f)


classes = set()
for item in o:
    classes.update(tuple(item['syms']))

classes = sorted(list(classes))

encs = {j:i+1 for i,j in enumerate(classes)}
os.makedirs(train_mask_path, exist_ok=True)
os.makedirs(test_mask_path, exist_ok=True)

for item in tqdm(o):
    img = Image.open(os.path.join(train_img_root, item['file_name']))
    shape = img.size
    if len(item['polygons']) > 0:
        mask = Image.new('L', (shape[1], shape[0]), 0)
        mask = ImageDraw.Draw(mask)
        for c, p in zip(item['syms'], item['polygons']):
            mask.polygon(tuple(map(tuple, p)),
                                 outline=encs[c],
                                 fill=encs[c])
        mask._image.save(os.path.join(train_mask_path, item['file_name']))


with open('ChestX_Det_test.json') as f:
    o = json.load(f)
    
for item in tqdm(o):
    img = Image.open(os.path.join(test_img_root, item['file_name']))
    shape = img.size
    if len(item['polygons']) > 0:
        mask = Image.new('L', (shape[1], shape[0]), 0)
        mask = ImageDraw.Draw(mask)
        for c, p in zip(item['syms'], item['polygons']):
            mask.polygon(tuple(map(tuple, p)), 
                                 outline=encs[c], 
                                 fill=encs[c]) 
        mask._image.save(os.path.join(test_mask_path, item['file_name']))
