import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

class Checkpoint():
    def __init__(self, epoch, model_path, model=None, optimizer=None, scheduler=None):
        self.epoch = epoch
        self.model_path = model_path
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._state = {'epoch': epoch,
                        'model': model,
                        'optimizer': optimizer,
                        'scheduler': scheduler}

    @property
    def epoch(self):
        return self._epoch

    @epoch.setter
    def epoch(self, val):
        if val<0:
            raise ValueError('Invalid epoch value: {}'.format(val))
        if not isinstance(val, int):
            raise TypeError('Epoch number must be int, but got {}.'.format(type(val)))
        self._epoch = val

    @property
    def model_path(self):
        return self._model_path

    @model_path.setter
    def model_path(self, val):
        if not isinstance(val, str):
            raise TypeError('Model path must be str, but got {}.'.format(type(val)))
        self._model_path = val

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, val):
        if val and not isinstance(val, nn.Module):
            raise TypeError('Model must be nn.Module, but got {}.'.format(type(val)))
        self._model = val

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, val):
        if val and not isinstance(val, optim.Optimizer):
            raise TypeError('Optimizer must be optim.Optimizer, but got {}.'.format(type(val)))
        self._optimizer = val

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, val):
        if val and not isinstance(val, optim.lr_scheduler._LRScheduler) and \
            not isinstance(val, optim.lr_scheduler.ReduceLROnPlateau):
            raise TypeError('Scheduler must be optim.lr_scheduler._LRScheduler, but got {}.'.format(type(val)))
        self._scheduler = val

    def save(self, val_metric=0):
        if self.model: self._state['model'] = self.model.state_dict()
        if self.optimizer: self._state['optimizer'] = self.optimizer.state_dict()
        if self.scheduler: self._state['scheduler'] = self.scheduler.state_dict()
        val_metric_str = '%.3f'%(val_metric)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        model_path = os.path.join(self.model_path,
                    'model_{}_e{}_val_{}.pt'.format(time.strftime("%Y%m%d-%H%M%S"), \
                    (str(self.epoch)), val_metric_str))
        torch.save(self._state, model_path)

    def load(self, ckpt, device="cpu"):
        ckpt_path = os.path.join(self.model_path, ckpt)
        if not os.path.exists(ckpt_path):
            raise ValueError('Checkpoint file does not exist: {}'.format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location=device)
        return ckpt_dict

def save_checkpoint(epoch, model_path, model=None, optimizer=None, scheduler=None, val_metric=0):
    ckpt = Checkpoint(epoch=epoch, model_path=model_path, model=model, optimizer=optimizer,scheduler=scheduler)
    ckpt.save(val_metric=val_metric)

def load_checkpoint(model_path, ckpt_name, device, model=None, optimizer=None, scheduler=None):
    ckpt = Checkpoint(epoch=0, model_path=model_path)
    ckpt_dict = ckpt.load(ckpt_name, device)
    start_epoch = ckpt_dict['epoch'] + 1
    if model and ckpt_dict['model']:
        model.load_state_dict(ckpt_dict['model'])
    if optimizer and ckpt_dict['optimizer']:
        optimizer.load_state_dict(ckpt_dict['optimizer'])
    if scheduler and ckpt_dict['scheduler']:
        scheduler.load_state_dict(ckpt_dict['scheduler'])
    return start_epoch, model, optimizer, scheduler
