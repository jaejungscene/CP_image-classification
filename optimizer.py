import math
from torch.optim.lr_scheduler import _LRScheduler
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

from torch.optim import SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, StepLR, ExponentialLR, \
    LambdaLR, SequentialLR, OneCycleLR


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr



def get_optimizer_and_scheduler(model, args, iter_per_epoch):
    """get optimizer and scheduler
    :arg
        model: nn.Module instance
        args: argparse instance containing optimizer and scheduler hyperparameter
    """
    parameter = model.parameters()
    total_iter = args.epoch * iter_per_epoch
    warmup_iter = args.warmup_epoch * iter_per_epoch

    if args.optimizer == 'sgd':
        optimizer = SGD(parameter, args.lr, args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(parameter, args.lr, eps=args.eps)
    elif args.optimizer == 'rmsprop':
        optimizer = RMSprop(parameter, args.lr, eps=args.eps, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        NotImplementedError(f"{args.optimizer} is not supported yet")

    if args.scheduler == 'cosine':
        main_scheduler = CosineAnnealingLR(optimizer, total_iter-warmup_iter, args.min_lr)
    elif args.scheduler == 'cosinerestarts':
        main_scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.epoch*iter_per_epoch//args.cosine_freq, T_mult=1, eta_max=args.eta_max, T_up=3, gamma=0.7)
    elif args.scheduler == 'multistep':
        main_scheduler = MultiStepLR(optimizer, [epoch * args.iter_per_epoch for epoch in args.milestones])
    elif args.scheduler == 'step':
        main_scheduler = StepLR(optimizer, total_iter-warmup_iter, gamma=args.decay_rate)
    elif args.scheduler =='explr':
        main_scheduler = ExponentialLR(optimizer, gamma=args.decay_rate)
    elif args.scheduler == 'onecyclelr':
        main_scheduler = OneCycleLR(optimizer, args.lr, total_iter, three_phase=args.three_phase)
    else:
        NotImplementedError(f"{args.scheduler} is not supported yet")

    if args.warmup_epoch and (args.scheduler != 'onecyclelr' and args.scheduler != "cosinerestarts"):
        if args.warmup_scheduler == 'linear':
            lr_lambda = lambda e: (e * (args.lr - args.warmup_lr) / warmup_iter + args.warmup_lr) / args.lr
            warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            NotImplementedError(f"{args.warmup_scheduler} is not supported yet")
        scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [warmup_iter])
    else:
        scheduler = main_scheduler

    return optimizer, scheduler