from fastai.vision.all import *
import timm
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from typing import list
from sklearn.model_selection import train_test_split
import kornia as K
import warnings
import pandas as pd
from augs import *
import os
from itertools import permutations
import random

class ClassificationDataset(Dataset):
    def __init__(self, xs, ys, augs):
        self.xs = xs
        self.ys = ys.to_numpy()
        self.augs = augs

    def __getitem__(self, idx):
        img = np.array(Image.open(self.xs[idx]).convert('L'))
        img = self.augs(image=img)['image']
        return img, torch.tensor(self.ys[idx], dtype=torch.long)

    def __len__(self): return len(self.xs)


def val_augs(img_sz):
    return A.Compose([A.Resize(img_sz, img_sz, p=1),
                      ToTensorV2()])


def get_model(name, c, in_chans=3):
    return nn.Sequential(*list(timm.create_model(name, pretrained=True, num_classes=c, in_chans=in_chans).children()))


class ImgProcCB(Callback):
    order = 10
    def before_batch(self):
        xb = K.enhance.normalize(self.xb[0]/255.0, 
                  mean=torch.tensor((0.485)), 
                  std=torch.tensor((0.229)))
        self.learn.xb = (xb,)


def train_split(k, c):
    os.makedirs('logs', exist_ok=True)
    df = pd.read_csv('train.csv')
    df['path'] = 'images_' + df['fold'].astype(str) + '/' + df['Image Index']

    train_df = df[(df['fold'] == c[0]) |
                  (df['fold'] == c[1]) |
                  (df['fold'] == c[2])].reset_index(drop=True)
    val_df = df[df['fold'] == c[3]].reset_index(drop=True)

    train_ds = ClassificationDataset(train_df['path'], train_df.iloc[:, 5:], get_medium_augmentations(512))
    val_ds = ClassificationDataset(val_df['path'], val_df.iloc[:, 5:], val_augmentations(512))

    dls = DataLoaders.from_dsets(train_ds, val_ds, batch_size=64).cuda()
    model = get_model('convnextv2_large', 15, in_chans=1).cuda()
    learn = Learner(dls, model, splitter=default_split, 
                                    metrics=[accuracy_multi], 
                                    loss_func=BCEWithLogitsLossFlat(), cbs=[ImgProcCB()]).to_fp16()
    learn.freeze()
    lr = learn.lr_find(show_plot=False)
    learn.fine_tune(24, base_lr=lr.valley, freeze_epochs=2, 
                                    cbs=[
                                        CSVLogger(fname=f'logs_{fold}.csv', append=False),
                                        #                     MixUp(),
                                        EarlyStoppingCallback(monitor='accuracy_multi', comp=np.greater, patience=4),
                                        GradientClip(1.0)],)
    torch.save(model.state_dict(), f'models/model_{fold}.pth')


for k in range(10):
    perms = list(permutations(range(5), 4))
    c = random.choice(perms)
    print(c)
    train_split(k, c)
