from fastai.vision.all import *
import fastprogress
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2, json, ast, urllib
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch import optim
import kornia as K
from map_boxes import mean_average_precision_for_boxes
# import xml.etree.ElementTree as ET
from torch.cuda.amp import autocast
import torchvision
from tqdm import tqdm
import timm
import string

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import warnings
warnings.filterwarnings('ignore')

IMG_SZ = 512
arch = 'convnextv2_tiny'

def get_train_tfms(img_sz):
    tfms = A.Compose([
      A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
             border_mode=cv2.BORDER_REFLECT),
         A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9)], p=0.9),
         A.HorizontalFlip(p=0.5),
         A.VerticalFlip(p=0.5),
         A.Resize(height=img_sz, width=img_sz, p=1.0),
         A.Cutout(num_holes = 8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
         ToTensorV2(p=1.0), ], p=1.0,
         bbox_params = A.BboxParams(format= 'pascal_voc',
                                   min_area=0,
                                   min_visibility=0,
                                   label_fields=['labels'])
        )
    return tfms

def get_valid_tfms(img_sz):
    tfms = A.Compose([A.Resize(height=img_sz, width=img_sz, p=1.0),
                     ToTensorV2(p=1.0)], p=1.0,
                    bbox_params = A.BboxParams(format='pascal_voc',
                                              min_area=0,
                                              min_visibility=0,
                                              label_fields=['labels']))
    return tfms


def xcycwh2xyxy(o):
    xc, yc, w, h = np.split(o, 4, axis=1)
    x1 = xc - w/2
    x2 = xc + w/2
    y1 = yc - h/2
    y2 = yc + h/2
    return np.concatenate([x1, y1, x2, y2], axis=1)

def clip_bbox(df, w, h):
    df['x1'].clip(0, w-1, inplace=True)
    df['x2'].clip(0, w, inplace=True)
    df['y1'].clip(0, h-1, inplace=True)
    df['y2'].clip(0, h, inplace=True)

def xyxy2area(o):
    x1, y1, x2, y2 = np.split(o, 4, axis=1)
    w = x2 - x1
    h = y2 - y1
    return w * h

class RCNNDataset():
    def __init__(self, fs, targs, enc, augs):
        self.fs = fs
        self.targs = targs
        self.encs = enc
        self.rev_encs = {j:i for i,j in enc.items()}
        self.augs = augs   
    
    def __len__(self): return len(self.fs)
    
    def __getitem__(self, idx):
        name = self.fs[idx]
        img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        targs = self.targs.loc[self.targs['file'] == name, :].reset_index(drop=True)
        # targs = self.targs.loc[self.targs['image_id']==idx, :].reset_index(drop=True)
        bboxes = clip_bbox(targs, *img.shape[:-1])
        if self.augs:
            temp = {}
            bboxes = targs.loc[:, 'x1,y1,x2,y2'.split(',')]
            sample = {'image': img,
                      'bboxes': targs.loc[:, 'x1,y1,x2,y2'.split(',')].values, 
                      'labels': targs.loc[:, 'category_id']}
            sample = self.augs(**sample)
            img = sample['image']
            if sample['bboxes']!=[]:
                temp['boxes'] = torch.stack(tuple(map(partial(torch.tensor, dtype=torch.float32),
                                                                zip(*sample['bboxes'])))).permute(1,0).float()
                temp['labels'] = torch.tensor(sample['labels']).long()
                if torch.equal(temp['boxes'], torch.tensor([[1., 1., 1., 1.]])):
                    temp['boxes'] == torch.tensor([[1., 1., 2., 2.]]).float()
                    temp['labels'] = torch.zeros(1).long()
                temp['area'] = xyxy2area(temp['boxes'])
                temp['iscrowd'] = torch.zeros_like(temp['area'], dtype=torch.int64)
            else: return img, {'boxes': torch.tensor([[1, 1, 2, 2]]).float(), 
                               'labels': torch.zeros(1).long(), 
                               'area': torch.ones(1),
                               'iscrowd': torch.zeros(1)}
        return img, temp


def process_dataset(annot_path, tfms):
    with open(annot_path) as f_read:
        o = json.load(f_read)
    temp = pd.DataFrame.from_records(o['annotations'])
    temp['x'] = temp['bbox'].map(lambda x: x[0]).astype(np.int32)
    temp['y'] = temp['bbox'].map(lambda x: x[1]).astype(np.int32)
    temp['w'] = temp['bbox'].map(lambda x: x[2]).astype(np.int32)
    temp['h'] = temp['bbox'].map(lambda x: x[3]).astype(np.int32)
    enc = {x['id']:x['name'] for x in o['categories']}
    temp['label'] = temp['category_id'].map(lambda x: enc[x])
    temp['x1'] = temp['x']
    temp['x2'] = temp['x'] + temp['w']
    temp['y1'] = temp['y']
    temp['y2'] = temp['y'] + temp['h']
    temp['area'] = temp['w'] * temp['h']
    temp = temp.drop(columns=['id', 'iscrowd', 'width', 'height', 'bbox', 'x', 'y', 'w', 'h'])
    temp = temp[temp['area'] > 20].reset_index(drop=True)
    f_map = {x['id']: x['file_name'] for x in o['images']}
    temp['file'] = temp['image_id'].map(lambda x: f_map[x])
    return RCNNDataset(temp['file'].unique().tolist(),
                        temp,
                        enc,
                        tfms)


def collate_fn(batch):
    return tuple(zip(*batch))

def proc_pred(image_id, train_ds, d):
    temp = pd.DataFrame(columns='ImageID,LabelName,Conf,XMin,XMax,YMin,YMax'.split(','))
    temp['XMin,XMax,YMin,YMax'.split(',')] = d['boxes']
    temp['LabelName'] = list(map(lambda x: train_ds.encs[x.item()], d['labels']))
    temp['Conf'] = d['scores'].detach().cpu().numpy()
    temp['ImageID'] = image_id
    return temp

def proc_targ(image_id, train_ds, d):
    temp = pd.DataFrame(columns='ImageID,LabelName,XMin,XMax,YMin,YMax'.split(','))
    temp['XMin,XMax,YMin,YMax'.split(',')] = d['boxes'].cpu().numpy()
    temp['LabelName'] = list(map(lambda x: train_ds.encs[x.item()], d['labels'].cpu().numpy()))
    temp['ImageID'] = image_id
    return temp


def train_loop(model, opt, dl, sched):
    model.train()
    loss_acc = []
    for k, (images, targ) in enumerate(tqdm(dl)):
        images = [image.cuda()/255.0 for image in images]
        targ = [{k: v.cuda() for k,v in t.items()} for t in targ]
        with autocast():
            loss_dict = model(images, targ)
            loss = sum(loss for loss in loss_dict.values())
            loss_acc.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        opt.zero_grad()
    return np.array(loss_acc).mean()

rand_string = lambda x: ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(x))

def val_loop(model, dl, train_ds):
    pred_acc, targ_acc = [], []
    model.eval()
    with torch.no_grad():
        for k, (images, targ) in enumerate(tqdm(dl)):
            images = [image.cuda()/255.0 for image in images]
            output = model(images)
            for i, (pred, targ) in enumerate(zip(output, targ)):
                pred_acc.append(proc_pred(str(k*dl.batch_size+i), train_ds, pred))
                targ_acc.append(proc_targ(str(k*dl.batch_size+i), train_ds, targ))
    
    targ_df = pd.concat(targ_acc, ignore_index=True)
    targ_df = targ_df[targ_df['LabelName'] != 'none']
    pred_df = pd.concat(pred_acc, ignore_index=True)
    pred_df = pred_df[pred_df['LabelName'] != 'none']
    rid = rand_string(5)
    print(rid)
    targ_df.to_csv(f'target_{rid}.csv', index=False)
    pred_df.to_csv(f'pred_{rid}.csv', index=False)
    mean_ap, average_precisions = mean_average_precision_for_boxes(targ_df.values, pred_df.values, iou_threshold=0.4)
    return mean_ap


def apply_mod(m, f):
    f(m)
    for l in m.children(): apply_mod(l, f)

def set_grad(m, b):
    if isinstance(m, (nn.Linear,nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)

def run_one_cycle(model, opt, lr, epochs, train_dl, val_dl, train_ds):
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=len(train_dl), epochs=epochs)
    for epoch in range(epochs):
        train_loss = train_loop(model, opt, train_dl, sched)
        print('train_loss : ', train_loss)
        val_loss = val_loop(model, val_dl, train_ds)

def resnet_param_groups(model):
    layers = []
    layers += [nn.Sequential(model.conv1, model.bn1)]
    layers += [l for name, l in model.named_children() if name.startswith("layer")]

    param_groups = [list(layer.parameters()) for layer in layers]
    check_all_model_params_in_groups2(model, param_groups)

    return param_groups

def create_model(arch, num_classes=1, pretrained=True):
    # Load the pretrained features.
    backbone = timm.create_model(arch, pretrained=True)
    modules = list(backbone.named_children())[:-1]
    features = nn.Sequential(OrderedDict(modules))

    features.out_channels = backbone.feature_info[-1]['num_chs']
    features.param_groups = MethodType(resnet_param_groups, features)

    anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
            )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
            )

    model = FasterRCNN(
        backbone=features,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model


def train_rcnn():
    train_ds = process_dataset('/home/npattab1/combined_dataset_train.json', get_train_tfms(IMG_SZ))
    val_ds = process_dataset('/home/npattab1/combined_dataset_val.json', get_valid_tfms(IMG_SZ))

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

    #train_ds, val_ds = RCNNDataset.from_coco(circuit_dir)
    # train_dl = torch.utils.data.DataLoader(train_ds, batch_size=8,
     #        shuffle=True, pin_memory=True, num_workers=4, 
     #        collate_fn=collate_fn, drop_last=True)
     #val_dl = torch.utils.data.DataLoader(val_ds, batch_size=8, 
     #        pin_memory=True, num_workers=4, collate_fn=collate_fn)
    model = create_model(arch, num_classes=len(train_ds.encs))
    model = model.cuda()
    # model = get_faster_rcnn(train_ds).cuda()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    # apply_mod(model.backbone, partial(set_grad, b=False))
    # run_one_cycle(model, opt, 1e-3, 3, train_dl, val_dl, train_ds)
    apply_mod(model.backbone, partial(set_grad, b=True))
    run_one_cycle(model, opt, 1e-4, 32, train_dl, val_dl, train_ds)


train_rcnn()
