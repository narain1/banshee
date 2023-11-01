import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from tqdm import tqdm
import torch
from fastai.vision.all import *
from sklearn.model_selection import train_test_split
import wandb
from fastai.callback.wandb import *


wandb.init(project='chestxdet')

with open('ChestX_Det_train.json') as f:
    o = json.load(f)

img_root = '/scratch/npattab1/segmentation/data4/train_data/train'
train_masks = '/scratch/npattab1/segmentation/data4/train_masks'
mean = [0.65459856,0.48386562,0.69428385]
std = [0.15167958,0.23584107,0.13146145]

classes = set()
for item in o:
    classes.update(tuple(item['syms']))

classes = sorted(list(classes))
encs = {j:i+1 for i,j in enumerate(classes)}

def get_aug(img_sz, p=1.0):
    return A.Compose([
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
            border_mode=cv2.BORDER_REFLECT),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=.1),
                A.PiecewiseAffine(p=0.3),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(10,15,10),
                A.CLAHE(clip_limit=2),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Resize(img_sz, img_sz),
            A.Normalize(mean=mean, std=std)
        ], p=p)

def val_aug(img_sz):
    return A.Compose([
        A.Resize(img_sz, img_sz),
        A.Normalize(mean=mean, std=std)
    ])

# Fix fastai bug to enable fp16 training with dictionaries

import torch
from fastai.vision.all import *
def flatten(o):
    "Concatenate all collections and items as a generator"
    for item in o:
        if isinstance(o, dict): yield o[item]; continue
        elif isinstance(item, str): yield item; continue
        try: yield from flatten(item)
        except TypeError: yield item

from torch.cuda.amp import GradScaler, autocast
@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs = kwargs
    def before_fit(self): 
        self.autocast,self.learn.scaler,self.scales = autocast(),GradScaler(**self.kwargs),L()
    def before_batch(self): self.autocast.__enter__()
    def after_pred(self):
        if next(flatten(self.pred)).dtype==torch.float16: self.learn.pred = to_float(self.pred)
    def after_loss(self): self.autocast.__exit__(None, None, None)
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)
    def before_step(self):
        "Use `self` as a fake optimizer. `self.skipped` will be set to True `after_step` if gradients overflow. "
        self.skipped=True
        self.scaler.step(self)
        if self.skipped: raise CancelStepException()
        self.scales.append(self.scaler.get_scale())
    def after_step(self): self.learn.scaler.update()

    @property 
    def param_groups(self): 
        "Pretend to be an optimizer for `GradScaler`"
        return self.opt.param_groups
    def step(self, *args, **kwargs): 
        "Fake optimizer step to detect whether this batch was skipped from `GradScaler`"
        self.skipped=False
    def after_fit(self): self.autocast,self.learn.scaler,self.scales = None,None,None
        
import fastai
fastai.callback.fp16.MixedPrecision = MixedPrecision


class FPN(nn.Module):
    def __init__(self, input_channels:list, output_channels:list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
             nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch*2),
             nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
            for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs:list, last_layer):
        hcs = [F.interpolate(c(x),scale_factor=2**(len(self.convs)-i),mode='bilinear')
               for i,(c,x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)

class UnetBlock(Module):
    def __init__(self, up_in_c:int, x_in_c:int, nf:int=None, blur:bool=False,
                 self_attention:bool=False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = nf if nf is not None else max(up_in_c//2,32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None,
            xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in:Tensor, left_in:Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
            [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d,groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                        nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                        nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False),
                                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class UneXt50(nn.Module):
    def __init__(self, stride=1, **kwargs):
        super().__init__()
        #encoder
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext50_32x4d_ssl')
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
                            m.layer1) #256
        self.enc2 = m.layer2 #512
        self.enc3 = m.layer3 #1024
        self.enc4 = m.layer4 #2048
        #aspp with customized dilatations
        self.aspp = ASPP(2048,256,out_c=512,dilations=[stride*1,stride*2,stride*3,stride*4])
        self.drop_aspp = nn.Dropout2d(0.5)
        #decoder
        self.dec4 = UnetBlock(512,1024,256)
        self.dec3 = UnetBlock(256,512,128)
        self.dec2 = UnetBlock(128,256,64)
        self.dec1 = UnetBlock(64,64,32)
        self.fpn = FPN([512,256,128,64],[16]*4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(32+16*4, 13, ks=1, norm_type=None, act_cls=None)
        self.cls = create_head(2048, 14)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        dec3 = self.dec4(self.drop_aspp(enc5),enc3)
        dec2 = self.dec3(dec3,enc2)
        dec1 = self.dec2(dec2,enc1)
        dec0 = self.dec1(dec1,enc0)
        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.interpolate(x,scale_factor=2,mode='bilinear')
        cls_out = self.cls(enc4)
        return x, cls_out

split_layers = lambda m: [list(m.enc0.parameters())+list(m.enc1.parameters())+
                list(m.enc2.parameters())+list(m.enc3.parameters())+
                list(m.enc4.parameters()),
                list(m.aspp.parameters())+list(m.dec4.parameters())+
                list(m.dec3.parameters())+list(m.dec2.parameters())+
                list(m.dec1.parameters())+list(m.fpn.parameters())+
                list(m.final_conv.parameters()) + list(m.cls.parameters())]


class SegmentationDataset(Dataset):
    def __init__(self, fs, classes, augs):
        self.fs = fs
        self.classes = classes
        self.augs = augs
        self.n_classes = len(classes)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(img_root, self.fs[idx])).convert('RGB')
        mask_path = os.path.join(train_masks, self.fs[idx])
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
            uniqs = np.eye(self.n_classes + 1, dtype=np.float32)[np.unique(mask)].sum(axis=0)
            uniqs[0] = 0
        else:
            mask = np.ones(img.size) * -1
            uniqs = np.zeros(14, dtype=np.float32)
            uniqs[0] = 1
        samp = {'image': np.array(img),
                'mask': np.array(mask)}
        samp = self.augs(**samp)
        img = samp['image']
        mask = samp['mask']
        _mask = np.zeros((mask.shape[0], mask.shape[1], len(self.classes)), dtype=np.uint8)
        _mask[:,:,np.arange(self.n_classes)] = (mask[:,:,np.newaxis] == np.arange(self.n_classes))
        img = torch.from_numpy(img).permute(2, 0, 1)
        targets = {'mask': torch.from_numpy(_mask).permute(2,0,1),
                   'targets': uniqs
                  }
        return img, targets

    def __len__(self): return len(self.fs)


class Dice_soft(Metric):
    def __init__(self, axis=1): 
        self.axis = axis 
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        pred,targ = flatten_check(torch.sigmoid(learn.pred[0]), learn.y['mask'])
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None

def loss_fn(pred, target):
    seg_loss = F.binary_cross_entropy_with_logits(pred[0], target['mask'].float(), reduction='none')
    s = seg_loss.shape
    seg_loss = torch.where(target['mask'] != -1, 
                           seg_loss, 
                           torch.zeros_like(seg_loss))
    cls_loss = F.binary_cross_entropy_with_logits(pred[1], target['targets'].float())
    return seg_loss.mean() + cls_loss


train_fs, val_fs = train_test_split(sorted(os.listdir(img_root)), random_state=32, shuffle=True)

train_ds = SegmentationDataset(train_fs, classes, get_aug(512))
val_ds = SegmentationDataset(val_fs, classes, val_aug(512))

dls = DataLoaders.from_dsets(train_ds, val_ds, batch_size=32, num_workers=4, pin_memory=True)

model = UneXt50().cuda()

learn = Learner(dls, model, loss_func=loss_fn,
                metrics=[Dice_soft()],
                cbs=[GradientClip(1.0),
                    WandbCallback()],
                splitter=split_layers).to_fp16()

learn.freeze_to(-1) #doesn't work
for param in learn.opt.param_groups[0]['params']:
    param.requires_grad = False

learn.fit_one_cycle(4, lr_max=1e-3)

learn.unfreeze()
learn.fit_one_cycle(32, lr_max=slice(1e-4, 6e-4),
        cbs=[SaveModelCallback(monitor='dice_soft',
            comp=np.greater),
            ])
