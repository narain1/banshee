import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import random
from fastai.vision.all import *
import timm
import segmentation_models_pytorch as smp
import albumentations as A
from sklearn.model_selection import train_test_split
import kornia as K
from torch import nn
import torch
from itertools import chain
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


def get_aug(img_sz, p=1.0):
    return A.Compose([
            A.Resize(img_sz, img_sz),
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
        ], p=p)

def val_aug(img_sz):
    return A.Resize(img_sz, img_sz)


class SegmentationDataset:
    def __init__(self, fs, augs):
        self.fs = fs
        self.augs = augs

    def __getitem__(self, idx):
        img = np.array(Image.open(os.path.join(train_path, self.fs[idx] + '.png')).convert('RGB'))
        mask = np.array(Image.open(os.path.join(mask_path, self.fs[idx] + '.png')).convert('L'))
        samp = {'image': img, 'mask': mask}
        samp = self.augs(**samp)
        img, mask = samp['image'], samp['mask']
        img, mask = torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(mask)
        return img, mask

    def __len__(self):
        return len(self.fs)


class ImgProcCB(Callback):
    def before_batch(self):
        xb = K.enhance.normalize(self.xb[0]/255.0,
            mean=torch.tensor((0.485, 0.456, 0.406)),
            std=torch.tensor((0.229, 0.224, 0.225)))
        self.learn.xb = (xb,)
        self.learn.yb = [self.yb[0].unsqueeze(1).float()]



class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels,
                kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                        for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                            * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
                                nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
                                nn.BatchNorm2d(fpn_out),
                                nn.ReLU(inplace=True)
        )

    def forward(self, features):

        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]##
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class UperNet_swin(nn.Module):
    # Implementing only the object path
    def __init__(self,model_name, img_size=256, num_classes=1, in_channels=3, pretrained=True):
        super(UperNet_swin, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, img_size=img_size)
        feature_channels = [o['num_chs'] for o in self.backbone.feature_info]
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        features = self.backbone(x)
        for idx in range(len(features)):
            features[idx] = features[idx].permute(0, 3, 1, 2)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode='bilinear')
        return x


class Dice_soft(Metric):
    def __init__(self, axis=1):
        self.axis = axis
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, learn):
        pred,targ = flatten_check(torch.sigmoid(learn.pred), learn.y)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return dice


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard



def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou



def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)



def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    #loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    loss = torch.dot(F.elu(errors_sorted)+1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss



def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))

