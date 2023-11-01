from pytorch_lightning.callbacks import Callback
from torch.distributions.beta import Beta

class Cutmix(Callback):
    def __init__(self, alpha=0.3):
        self.distrib = Beta(tensor(alpha), tensor(alpha))

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        xb, yb = batch
        w, h = self.xb.size(3), self.xb.size(2)
        lam = self.distrib.sample((1,)).squeeze().to(self.xb.device)
        self.lam = lam.max()
        shuffle = torch.randperm(self.y.size(0)).to(self.xb.device)
        xb1, yb1 = xb[shuffle], yb[shuffle]
        n_dims = len(self.xb.size())
        x1, y1, x2, y2 = self.rand_bbox(w, h, self.lam)
        xb[:,:,x1:x2, y1:y2] = xb1[:, :, x1:x2, y1:y2]
        self.lam = (1 - ((x2 - x1) *(y2-y1))/float(w*h)).item()

    @staticmethod
    def lf(self, pred, yb1, yb2):
        loss = torch.lerp(self.loss_fn(pred, yb1), self.loss_fn(pred, yb2), self.lam)
        return loss

    def rand_bbox(self, w, h, lam):
        cut_rat = torch.sqrt(1. - lam)
        cut_w = (w * cut_rat).type(torch.long)
        cut_h = (h * cut_rat).type(torch.long)
        cx = torch.randint(0, w, (1,)).to(self.xb.device)
        cy = torch.randint(0, h, (1,)).to(self.xb.device)
        x1 = torch.clamp(cx - cut_w // 2, 0, w)
        y1 = torch.clamp(cy - cut_h // 2, 0, h)
        x2 = torch.clamp(cx + cut_w // 2, 0, w)
        y2 = torch.clamp(cy + cut_h // 2, 0, h)
        return x1, y1, x2, y2


from fastai.callback import Callback
from sklearn.metrics import roc_auc_score
import numpy as np
import torch

class MultiAUC(Callback):
    def __init__(self, average='macro', flatten=True):
        super().__init__()
        self.average = average
        self.flatten = flatten
        self.metric_name = f'MultiAUC_{average}'

    def on_epoch_begin(self, **kwargs):
        self.outputs, self.targets = [], []

    def on_batch_end(self, last_output, last_target, **kwargs):
        # Flatten the inputs if required
        if self.flatten:
            last_output = last_output.view(-1, last_output.shape[-1])
            last_target = last_target.view(-1, last_target.shape[-1])

        # Convert tensors to numpy arrays
        last_output = last_output.detach().cpu().numpy()
        last_target = last_target.detach().cpu().numpy()

        # Append outputs and targets
        self.outputs.append(last_output)
        self.targets.append(last_target)

    def on_epoch_end(self, last_metrics, **kwargs):
        # Concatenate outputs and targets for all batches
        outputs = np.concatenate(self.outputs)
        targets = np.concatenate(self.targets)

        # Calculate the multi-AUC score
        auc_scores = []
        for class_index in range(outputs.shape[1]):
            auc = roc_auc_score(targets[:, class_index], outputs[:, class_index])
            auc_scores.append(auc)

        if self.average == 'macro':
            auc_score = np.mean(auc_scores)
        elif self.average == 'micro':
            auc_score = roc_auc_score(targets, outputs, average=self.average)
        else:
            raise ValueError(f"Unsupported average type: {self.average}")

        return add_metrics(last_metrics, auc_score)