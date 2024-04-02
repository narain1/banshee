# fastai integration of Accelerate

from accelerate import Accelerator

from fastai.callback.core import Callback, CancelBatchException, CancelStepException
from fastai.learner import Learner, Metric
from fastai.metrics import AccumMetric
from fastai.optimizer import Optimizer, _update
from fastai.distributed import DistributedDL
from fastai.torch_core import to_device
import torch
from torch.utils.data import DataLoader

from accelerate.optimizer import AcceleratedOptimizer
from fastcore.basics import patch

@patch
def __call__(self:AccumMetric, preds, targs):
    preds, targs = self.gather(preds, targs)
    self.reset()
    self.accum_values(preds, targs)
    return self.value

@patch
def gather(self:Metric, learn:Learner, y_preds:torch.Tensor, yb:torch.Tensor):
    """
    Gathers `y_preds` and `yb` across all devices

    Args:
        learn (`Learner`):
            A fastai `Learner`
        y_preds (`torch.Tensor`):
            Outputs from a torch model.
        yb (`torch.Tensor`):
            A batch of labels
    """
    return learn.accelerator.gather((y_preds, *yb))

# Make step be "compatible" with a closure argument
@patch
def step(self:Optimizer, closure=None):
    for p,pg,state,hyper in self.all_params(with_grad=True):
        for cb in self.cbs: state = _update(state, cb(p, **{**state, **hyper}))
        self.state[p] = state

@patch(as_prop=True)
def hypers(self:AcceleratedOptimizer):
    return self.optimizer.hypers

class AcceleratorCallback(Callback):
    def __init__(self, accelerator:Accelerator):
        """
        A Callback that handles preparing the model, dataloaders, and optimizer with Accelerate

        Args:
            accelerator (`Accelerator`):
                An instance of `Accelerator`, stored in `self.learn.accelerator`
        """
        self.accelerator = accelerator

    def before_fit(self):
        """
        Assing `self.accelerator` to `self.learn.accelerator` and prepare the model
        """
        self.learn.accelerator = self.accelerator
        self.learn.model = self.accelerator.prepare(self.learn.model)
        self.learn.opt = self.accelerator.prepare_optimizer(self.learn.opt)
        self.learn.accelerator._optimizers.append(self.learn.opt)

    @staticmethod
    def _prepare_dataloader(dataloader, accelerator):
        """
        Prepares a single dataloader with either Accelerate or DistributedDL
        """
        if isinstance(dataloader, DataLoader):
            return accelerator.prepare(dataloader)
        elif not isinstance(dataloader, DistributedDL):
            return DistributedDL(
                dataloader, 
                rank=accelerator.process_index, 
                world_size=accelerator.num_processes
            )

    def before_train(self):
        """
        Prepare the training dataloader
        """
        if self.accelerator.num_processes > 1:
            self.learn.dl = self._prepare_dataloader(self.learn.dl, self.accelerator)

    def before_validate(self):
        """
        Prepare the validation dataloader
        """
        if self.accelerator.num_processes > 1:
            self.learn.dl = self._prepare_dataloader(self.learn.dl, self.accelerator)

@patch
def _do_one_batch(self:Learner):
    self.pred = self.model(*self.xb)
    self('after_pred')
    if len(self.yb):
        self.loss_grad = self.loss_func(self.pred, *self.yb)
        self.loss = self.loss_grad.clone()
    self('after_loss')
    if not self.training or not len(self.yb):
        return 
    self('before_backward')
    if hasattr(self, 'accelerator'):
        self.accelerator.backward(self.loss_grad)
    else:
        self.loss_grad.backward(self.loss_grad)
    self._with_events(self.opt.step, 'step', CancelStepException)
    self.opt.zero_grad()

@patch
def _set_device(self:Learner, b):
    if hasattr(self, "accelerator"):
        return to_device(b, self.accelerator.device)
    else:
        model_device = torch.device(torch.cuda.current_device()) if next(self.model.parameters()).is_cuda else torch.device('cpu')
        dls_device = getattr(self.dls, 'device', default_device())
        if model_device == dls_device: return to_device(b, dls_device)
        else: return to_device(b, model_device)

@patch
def one_batch(self:Learner, i, b):
    self.iter = i
    b = self._set_device(b)
    self._split(b)
    self._with_events(self._do_one_batch, 'batch', CancelBatchException)


# accelerate launch train_fastai.py
