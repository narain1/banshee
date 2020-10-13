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
