import json
import argparse
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.model import Worker
from libs.utils import *


def main(args):
    ckpt_path = os.path.join('log', args.name)
    cfg_path = os.path.join(ckpt_path, 'config.yaml')
    check_file(cfg_path)
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
        split= ['test'],
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
    results_list = []
    # Rank @ n, IOU @ m
    n, m = (1, 5), (0.1, 0.3, 0.5, 0.7, 0.9)
    topk = max(n)
    for itr, data_list in enumerate(eval_loader, 1):
        results = worker.eval(data_list[0], cfg['eval'], ema=args.ema)
        segs, scores = results['segments'], results['scores']
        idx = scores.argsort(descending=True)
        segs, scores = segs[idx[:topk]], scores[idx[:topk]]
        preds = segs.tolist()
        temp_dict = {
            'clip_uid'          : data_list[0]['id'],
            'annotation_uid'    : data_list[0]['annotation_uid'],
            'query_idx'         : data_list[0]['query_idx'],
            'predicted_times'   : preds,
        }
        results_list.append(temp_dict)

    submit_content = {
        'version'           : '1.0',
        'challenge'         : 'ego4d_nlq_challenge',
        'results'           : results_list,
    }


    with open('results.json','w') as f:
        json.dump(submit_content,f)
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