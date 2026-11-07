import warnings

import torch
from torch import distributed as dist

import functools
import os
from collections import OrderedDict
from math import cos, pi
from .dist import get_dist_info, master_only
import torch.optim.lr_scheduler as lr_scheduler
import random
import numpy as np


def set_seed(seed):
    """
    Setting of Global Seed
    设置全局随机种子，保证实验可复现
    """
    np.random.seed(seed)      
    random.seed(seed)         
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed)   

     
     
     
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed=None):

    if seed is not None:
        random.seed(seed + worker_id)
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        torch.cuda.manual_seed(seed + worker_id)
        torch.cuda.manual_seed_all(seed + worker_id)


class AverageMeter(object):
    """Computes and stores the average and current value.计算和存储值"""
    def __init__(self, apply_dist_reduce=False):
        self.apply_dist_reduce = apply_dist_reduce
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def dist_reduce(self, val):
        """分布式训练评估值减少，其实就是评估值取均值"""
        rank, world_size = get_dist_info()   
        if world_size == 1:
            return val
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val, device="cuda")
        dist.all_reduce(val)
        return val.item() / world_size

    def get_val(self):
        """获取评估值"""
        if self.apply_dist_reduce:
            return self.dist_reduce(self.val)
        else:
            return self.val

    def get_avg(self):
        """获取平均值"""
        if self.apply_dist_reduce:
            return self.dist_reduce(self.avg)
        else:
            return self.avg

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


 
def cosine_lr_after_step(cfg, optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6, warm_epoch=0, warm_lr=2e-5):

    if cfg.decoder_multiplier == 1:
        if warm_epoch <= epoch < step_epoch:
            lr = base_lr
        elif epoch < warm_epoch:
            lr = warm_lr
        else:
             
            lr = clip + 0.5 * (base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))

         
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    else:
        for param_group in optimizer.param_groups:
             
             
            group_base_lr  = param_group.get('initial_lr', param_group["lr"])

            if warm_epoch <= epoch < step_epoch:
                lr = group_base_lr 
            elif epoch < warm_epoch:
                 
                ratio = group_base_lr / base_lr
                lr = warm_lr * ratio
            else:
                 
                lr = clip + 0.5 * (group_base_lr - clip) * (1 + cos(pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))

             
            param_group["lr"] = lr

    return optimizer.param_groups[0]["lr"]


def cadt_cosine_lr_after_step(optimizer, epoch, cfg, clip=1e-6):

    epoch_warmup = cfg.epoch_warmup
    lr_warmup = cfg.learning_rate_warmup
    total_epochs = cfg.epoch
    if epoch < epoch_warmup:
        lr = lr_warmup
    else:
         
        lr = clip + 0.5 * (cfg.learning_rate - clip) * (1 + cos(pi * ((epoch - epoch_warmup) / (total_epochs - epoch_warmup))))

     
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class multi_cosine_lr_after_step():
    def __init__(self, optimizer, base_lr, step_epoch, total_epochs, T_i=10, T_mult=2, clip=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.step_epoch = step_epoch
        self.total_epochs = total_epochs
        if T_i <= step_epoch:
            self.T_i = min(step_epoch + 5, total_epochs)   
            warnings.warn(f"T_i自动调整为{self.T_i} (原值{T_i}≤step_epoch)", UserWarning)
        else:
            self.T_i = min(T_i, total_epochs)
        self.T_mult = T_mult
        self.cur_T = step_epoch
        self.iter_step = 1
        self.clip = clip

    def step(self, epoch):
        if epoch < self.step_epoch:
            lr = self.base_lr
        else:
            if epoch - self.cur_T >= self.T_i:
                self.cur_T = epoch
                self.T_i = min(self.T_i*self.T_mult, self.total_epochs - epoch)
                self.iter_step += 1
             
            lr = self.clip + (0.5**self.iter_step) * (self.base_lr - self.clip) * (1 + cos(pi * ((epoch - self.cur_T) / (self.T_i - self.step_epoch))))

         
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def get_scheduler(cfg, optimizer):
    if cfg.type == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_decay_epochs, gamma=cfg.lr_decay)
    else:
        raise Exception('Not impl. such scheduler')
    return scheduler


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def weights_to_cpu(state_dict):
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.
    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    return state_dict_cpu


@master_only
def checkpoint_save(epoch, model, optimizer, work_dir, save_freq=16, best=False):

    if hasattr(model, "module"):
        model = model.module   

    if best:
        checkpoint = {"net": weights_to_cpu(model.state_dict()), "epoch": epoch}
        torch.save(checkpoint, f"{work_dir}/best.pth")   
        return

    f = os.path.join(work_dir, f"epoch_{epoch}.pth")
    checkpoint = {"net": weights_to_cpu(model.state_dict()), "optimizer": optimizer.state_dict(), "epoch": epoch}
    torch.save(checkpoint, f)

    f_latest = os.path.join(work_dir, "latest.pth")
    if os.path.exists(f_latest):
        os.remove(f"{work_dir}/latest.pth")
    torch.save(checkpoint, f_latest)   

     

     
    epoch = epoch - 1
    f = os.path.join(work_dir, f"epoch_{epoch}.pth")
    if os.path.isfile(f):
        if not is_multiple(epoch, save_freq) and not is_power2(epoch):
            os.remove(f)


def load_checkpoint(checkpoint, logger, model, optimizer=None, strict=False):
    if hasattr(model, "module"):
        model = model.module
    device = torch.cuda.current_device()
    state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage.cuda(device))
    src_state_dict = state_dict["net"]
    target_state_dict = model.state_dict()
    skip_keys = []
     
    for k in src_state_dict.keys():
        if k not in target_state_dict:
            continue
        if src_state_dict[k].size() != target_state_dict[k].size():
            skip_keys.append(k)
    for k in skip_keys:
        del src_state_dict[k]
    missing_keys, unexpected_keys = model.load_state_dict(src_state_dict, strict=strict)
    if skip_keys:
        logger.info(f'removed keys in source state_dict due to size mismatch: {", ".join(skip_keys)}')
    if missing_keys:
        logger.info(f'missing keys in source state_dict: {", ".join(missing_keys)}')
    if unexpected_keys:
        logger.info(f'unexpected key in source state_dict: {", ".join(unexpected_keys)}')

     
    if optimizer is not None:
        assert "optimizer" in state_dict
        optimizer.load_state_dict(state_dict["optimizer"])

    if "epoch" in state_dict:
        epoch = state_dict["epoch"]
    else:
        epoch = 0
    return epoch + 1


def get_max_memory():

    mem = torch.cuda.max_memory_allocated()
    mem_mb = torch.tensor([int(mem) // (1024 * 1024)], dtype=torch.int, device="cuda")
    _, world_size = get_dist_info()
    if world_size > 1:
        dist.reduce(mem_mb, 0, op=dist.ReduceOp.MAX)
    return mem_mb.item()


def cuda_cast(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for x in args:
            if isinstance(x, torch.Tensor):
                x = x.cuda()
            new_args.append(x)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.cuda()
            new_kwargs[k] = v
        return func(*new_args, **new_kwargs)

    return wrapper

def move_batch_to_device(batch, device):

    if isinstance(batch, torch.Tensor):
         
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(item, device) for item in batch)
    elif isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
     
    else:
        return batch
