import os
import time
import shutil
import random
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import numpy as np

_log_path = None


def set_gpu(gpu):
    print('set gpu: {:s}'.format(gpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def check_file(path):
    if not os.path.isfile(path):
        raise ValueError('file does not exist: {:s}'.format(path))


def check_path(path):
    if not os.path.exists(path):
        raise ValueError('path does not exist: {:s}'.format(path))


def ensure_path(path, remove=False):
    if os.path.exists(path):
        if remove:
            if input('{:s} exists, remove? ([y]/n): '.format(path)) != 'n':
                shutil.rmtree(path)
                os.makedirs(path)
    else:
        os.makedirs(path)


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def count_params(model, return_str=True):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    if return_str:
        if n_params >= 1e6:
            return '{:.1f}M'.format(n_params / 1e6)
        else:
            return '{:.1f}K'.format(n_params / 1e3)
    else:
        return n_params


class AverageMeter(object):

    def __init__(self):

        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.mean = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.mean = self.sum / self.count

    def item(self):
        return self.mean


class Timer(object):

    def __init__(self):

        self.start()

    def start(self):
        self.v = time.time()

    def end(self):
        return time.time() - self.v


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t > 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def fix_random_seed(seed, reproduce=False):
    cudnn.enabled = True
    cudnn.benchmark = True
    
    if reproduce:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ## NOTE: uncomment for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        ## NOTE: uncomment for pytorch >= 1.8
        # torch.use_deterministic_algorithms(True)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)

    return rng


def iou(pred_segs, gt_segs):
    """
    Args:
        pred_segs (float tensor, (..., 2)): predicted segments.
        gt_segs (float tensor, (..., 2)): ground-truth segments.

    Returns:
        out (float tensor, (...)): intersection over union.
    """
    ps, pe = pred_segs[..., 0], pred_segs[..., 1]
    gs, ge = gt_segs[..., 0], gt_segs[..., 1]
    
    overlap = (torch.minimum(pe, ge) - torch.maximum(ps, gs)).clamp(min=0)
    union = (pe - ps) + (ge - gs) - overlap
    out = overlap / union
    
    return out