import os
import random

import torch
import numpy as np


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing.
    """
    return batch


def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker.
    """
    seed = torch.initial_seed() % 2 ** 31
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def jitter_seg_bounds(
        data_dict,
        jitter=None,
        num_trials=1000,
):
    """
        Jitter segment bounds in a dict item.
    """
    if jitter is None:
        return

    ts, te = data_dict['target']
    seg_len = te - ts
    jitter_range = jitter * seg_len

    for _ in range(num_trials):
        left = ts + random.uniform(-jitter_range, jitter_range)
        right = te + random.uniform(-jitter_range, jitter_range)
        if right > left:
            data_dict['target'] = (left, right)
            return

    raise ValueError(
        (
            'invalid jitter: [vid] {:d}, [seg] ({:.2f}, {:.2f})'
            ''.format(
                data_dict['vid_feats'].size(-1),
                data_dict['target'][0], data_dict['target'][1]
            )
        )
    )

    return


def truncate_vid_feats(
        data_dict,
        max_seq_len,
        trunc_thresh=0.5,
        crop_ratio=None,
        num_trials=10000,
):
    temp_data_dict = data_dict
    """
        Truncate feats and timestamps in a dict item.
    """
    seq_len = data_dict['vid_feats'].size(1)
    ts, te = data_dict['target']

    # corner case 1: event starts before 0
    if ts < 0:
        if seq_len > max_seq_len:
            data_dict['vid_feats'] = data_dict['vid_feats'][:, :max_seq_len]
            left, right = ts, min(te, max_seq_len)
            data_dict['target'] = (left, right)
        return

    # corner case 2: event ends after seq_len
    if te > seq_len:
        if seq_len > max_seq_len:
            data_dict['vid_feats'] = data_dict['vid_feats'][:, -max_seq_len:]
            delta = seq_len - max_seq_len
            left, right = max(0, ts - delta), te - delta
            data_dict['target'] = (left, right)
        return

    if seq_len <= max_seq_len:
        if crop_ratio is None:
            return
        max_seq_len = random.randint(
            max(round(crop_ratio[0] * seq_len), 1),
            min(round(crop_ratio[1] * seq_len), seq_len)
        )
        if seq_len == max_seq_len:
            return

    for _ in range(num_trials):
        # randomly sample a window
        ws = random.randint(0, seq_len - max_seq_len)
        we = ws + max_seq_len

        # compute overlap between sampled window and target segment
        left, right = max(ws, ts), min(we, te)
        overlap = max(right - left, 0)

        # window with sufficient overlap has been found
        if overlap / (te - ts + 1e-5) > trunc_thresh:
            data_dict['vid_feats'] = data_dict['vid_feats'][:, ws:we]
            data_dict['target'] = (left - ws, right - ws)
            return


    raise ValueError(
        (
            'invalid truncation: [vid] {:d}, [seg] ({:.2f}, {:.2f})'
            ''.format(
                data_dict['vid_feats'].size(-1),
                data_dict['target'][0], data_dict['target'][1]
            )
        )
    )

    return