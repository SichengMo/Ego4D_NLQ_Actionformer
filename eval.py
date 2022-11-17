import os
import argparse

import torch
import numpy as np

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.model import Worker
from libs.utils import *


def main(args):
    ckpt_path = os.path.join('log', args.name)
    cfg_path = os.path.join(ckpt_path, 'config.yaml')
    check_file(cfg_path)
    set_log_path(ckpt_path)
    cfg = load_config(cfg_path)
    print('config loaded from checkpoint folder')
    set_gpu(args.gpu)

    ###########################################################################
    """ worker """

    ckpt_name = os.path.join(ckpt_path, '{:s}.pth'.format(args.ckpt))
    check_file(ckpt_name)
    ckpt = torch.load(ckpt_name)

    worker = Worker(cfg['model'])
    worker.load(ckpt)
    worker.cuda()
    print('worker initialized')

    ###########################################################################
    """ dataset """

    eval_set = make_dataset(
        name=cfg['data']['dataset']['name'],
        split=cfg['data']['eval_split'],
        cfg=cfg['data']['dataset'],
        is_training=False,
    )
    eval_loader = make_data_loader(
        eval_set,
        generator=None,
        batch_size=1,
        num_workers=1,
        is_training=False,
    )
    print('eval data size: {:d}'.format(len(eval_set)))

    ###########################################################################
    """ eval """

    # Rank @ n, IOU @ m
    n, m = (1, 5), (0.1, 0.3, 0.5, 0.7, 0.9)
    counts = np.zeros((len(n), len(m)))
    topk = max(n)
    m = np.array(m)

    for itr, data_list in enumerate(eval_loader, 1):
        results = worker.eval(data_list[0], cfg['eval'], ema=args.ema)

        segs, scores = results['segments'], results['scores']
        idx = scores.argsort(descending=True)
        segs, scores = segs[idx[:topk]], scores[idx[:topk]]
        gt = torch.as_tensor(data_list[0]['segment'], dtype=torch.float)
        gt = gt.expand(len(segs), -1)

        iou_topk = iou(segs, gt)
        iou_n = np.array([iou_topk[:i].max().item() for i in n])
        counts += iou_n[:, None] > m[None]

        if itr % 100 == 0:
            metrics = counts / itr
            print('\n[{:d}/{:d}]'.format(itr, len(eval_set)))
            for i in range(len(n)):
                print('\t-----')
                for j in range(len(m)):
                    print(
                        '\tRank@{:d}, IoU@{:.1f}: {:.2f}'.format(
                            n[i], m[j], metrics[i, j] * 100
                        )
                    )

    metrics = counts / len(eval_set)
    print('\nFinal:')
    log_str = ''
    for i in range(len(n)):
        log_str += '\n-----'
        for j in range(len(m)):
            log_str += (
                '\nRank@{:d}, IoU@{:.1f}: {:.2f}'.format(
                    n[i], m[j], metrics[i, j] * 100
                )
            )
    log(log_str,'results_{:s}.txt'.format(args.ckpt))

    ###########################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='job name')
    parser.add_argument('-c', '--ckpt', type=str, default='last',
                        help='checkpoint name')
    parser.add_argument('-ema', action='store_true', help='use EMA model')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU IDs')
    args = parser.parse_args()

    main(args)
