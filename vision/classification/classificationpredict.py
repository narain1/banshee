import timm
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from typing import List
from sklearn.model_selection import train_test_split
import Kornia as K
import warnings
import pandas as pd
from fastai.vision.all import *
from PIL import Image
import numpy as np
import os
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')


class ClassificationDataset(Dataset):
    def __init__(self, xs, ys, augs):
        self.xs = xs
        self.ys = ys
        self.augs = augs


    def __getitem__(self, idx):
        img = np.array(Image.open(self.xs[idx]).convert('L'))
        img = self.augs(image=img)['image']
        return img, torch.tensor(self.ys[idx], dtype=torch.long)

    def __len__(self):
        return len(self.xs)


def val_augs(img_sz):
    return A.Compose([A.Resize(img_sz, img_sz, p=1),
                      ToTensorV2()])


def get_model(name, c, in_chans=3):
    return nn.Sequential(*list(timm.create_model(name,
                                                 pretrained=False,
                                                 num_classes=c,
                                                 in_chans=in_chans).children()))

def load_model(p)
    model = get_model('convnextv2_base', 15, in_chans=1)
    model.load_state_dict(torch.load(p, map_location='cpu'))
    model = model.cuda()
    model = model.eval()
    return model


def predict(model):
    acc_pred, acc_targ = [], []
    with torch.no_grad():
        for k, (xb, yb) in enumerate(tqdm(test_dl)):
            xb, yb = xb.cuda(), yb.cuda()
            xb = K.enhance.normalize(xb.float()/255.0,
                                     mean=torch.tensor((0.485)),
                                     std=torch.tensor((0.229)))
            ys = model(xb)
            acc_pred.append(ys.sigmoid().detach().cpu().numpy())
            acc_targ.append(yb.detach().cpu().numpy())
    preds = np.concatenate(acc_pred)
    targs = np.concatenate(acc_targ)
    metrics = dict()
    for k, c in enumerate(cols):
        if k != 0:
            metrics[c] = roc_auc_score(targs[:, k].flatten(), preds[:, k].flatten())
    return metrics


test_df = pd.read_csv('test.csv')
test_df['path'] = 'test/' + test_df['Image Index']
cols = test_df.iloc[:, 4:].columns.tolist()


test_ds = ClassificationDataset(test_df['path'], test_df.iloc[:, 4:], val_augs(512))
test_dl = DataLoader(test_ds, batch_size=128, num_workers=8, pin_memory=True)



model_path = 'models'
model_list = [os.path.join(model_path, x) for x in os.listdir(model_path)]


acc = []
for f in model_list:
    m = load_model(p)
    d = {'name': os.path.basename(f)}
    out = predict(m)
    d.update(out)
    acc.append(d)

temp_df = pd.DataFrame.from_records(acc)
temp_df.to_csv('model_metrics.csv', index=False)
