import os
import argparse

import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.model import Worker
from libs.utils import *


def main(args):
    # set up checkpoint folder
    os.makedirs('log', exist_ok=True)
    ckpt_path = os.path.join('log', args.name)
    ensure_path(ckpt_path)

    # load config
    try:
        cfg_path = os.path.join(ckpt_path, 'config.yaml')
        check_file(cfg_path)
        cfg = load_config(cfg_path)
        print('config loaded from checkpoint folder')
        cfg['_resume'] = True
    except:
        check_file(args.config)
        cfg = load_config(args.config)
        print('config loaded from command line')

    # configure GPUs
    n_gpus = len(args.gpu.split(','))
    if n_gpus > 1:
        cfg['_parallel'] = True
        cfg['opt']['lr'] *= n_gpus  # linear scaling rule
    set_gpu(args.gpu)

    set_log_path(ckpt_path)
    writer = SummaryWriter(os.path.join(ckpt_path, 'tensorboard'))
    rng = fix_random_seed(cfg.get('seed', 2022))

    ###########################################################################
    """ worker """

    ep0 = 0
    if cfg.get('_resume'):
        ckpt_name = os.path.join(ckpt_path, 'last.pth')
        try:
            check_file(ckpt_name)
            ckpt = torch.load(ckpt_name)
            ep0, cfg = ckpt['epoch'], ckpt['config']
            worker = Worker(cfg['model'])
            worker.load(ckpt)
        except:
            cfg.pop('_resume')
            ep0 = 0
            worker = Worker(cfg['model'])
    else:
        worker = Worker(cfg['model'])
        yaml.dump(cfg, open(os.path.join(ckpt_path, 'config.yaml'), 'w'))

    worker.cuda(cfg.get('_parallel'))
    print('worker initialized, train from epoch {:d}'.format(ep0 + 1))
    print('number of model parameters: {:s}'.format(count_params(worker)))

    ###########################################################################
    """ dataset """

    train_set = make_dataset(
        name=cfg['data']['dataset']['name'],
        split=cfg['data']['train_split'],
        cfg=cfg['data']['dataset'],
        is_training=True,
    )
    train_loader = make_data_loader(
        train_set,
        generator=rng,
        batch_size=cfg['data']['batch_size'],
        num_workers=cfg['data']['num_workers'],
        is_training=True,
    )

    cfg['opt']['itrs_per_epoch'] = itrs_per_epoch = len(train_loader)
    print('train data size: {:d}'.format(len(train_set)))
    print('number of iterations per epoch: {:d}'.format(itrs_per_epoch))

    ###########################################################################
    """ optimizer & scheduler """

    optimizer = make_optimizer(worker, cfg['opt'])
    scheduler = make_scheduler(optimizer, cfg['opt'])

    n_epochs = cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    print(cfg['opt']['epochs'])
    n_itrs = n_epochs * itrs_per_epoch

    ###########################################################################
    """ train """

    loss_list = ['cls', 'reg', 'iou', 'nce', 'total']
    losses = {k: AverageMeter() for k in loss_list}
    timer = Timer()

    for ep in range(ep0, n_epochs):
        # train for one epoch
        for itr, data_list in enumerate(train_loader, 1):
            loss_dict = worker.train(data_list, cfg['train'])

            global_itr = ep * itrs_per_epoch + itr
            for k in loss_list:
                if k in loss_dict.keys():
                    losses[k].update(loss_dict[k].item())
                    writer.add_scalar(k, losses[k].item(), global_itr)
            lr = scheduler.get_last_lr()[0]
            writer.add_scalar('lr', lr, global_itr)
            writer.flush()

            optimizer.zero_grad()
            loss_dict['total'].backward()
            if cfg['opt']['clip_grad_norm'] > 0:
                nn.utils.clip_grad_norm_(
                    worker.parameters(), cfg['opt']['clip_grad_norm']
                )

            optimizer.step()
            scheduler.step()
            worker.ema_update(cfg['opt'].get('ema_momentum', 0.999))

            if global_itr == 1 or global_itr % args.print_freq == 0:
                torch.cuda.synchronize()
                t_elapsed = time_str(timer.end())

                log_str = '[{:03d}/{:03d}] '.format(
                    global_itr // args.print_freq, n_itrs // args.print_freq
                )
                for k in loss_list:
                    if k in loss_dict.keys():
                        log_str += '{:s} {:.3f} ({:.3f}) | '.format(
                            k, loss_dict[k].item(), losses[k].item()
                        )
                        losses[k].reset()
                log_str += t_elapsed
                log(log_str, 'log.txt')
                timer.start()

        # save checkpoint
        ckpt = worker.save()
        ckpt['epoch'] = ep + 1
        ckpt['config'] = cfg
        ckpt['optimizer'] = optimizer.state_dict()
        ckpt['scheduler'] = scheduler.state_dict()
        torch.save(ckpt, os.path.join(ckpt_path, '{:02d}.pth'.format(ep + 1)))
        torch.save(ckpt, os.path.join(ckpt_path, 'last.pth'))

    writer.close()
    print('all done!')

    ###########################################################################


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='config file path')
    parser.add_argument('-n', '--name', type=str, help='job name')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU IDs')
    parser.add_argument('-pf', '--print_freq', type=int, default=1,
                        help='print frequency (x100 itrs)')
    args = parser.parse_args()
    args.print_freq *= 100

    main(args)