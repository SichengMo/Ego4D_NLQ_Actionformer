import yaml


DEFAULTS = {
    'seed': 1234567891,

    'data': {
        'train_split': ('train',),
        'eval_split': ('val',),
        
        'batch_size': 16,
        'num_workers': 8,
        
        'dataset': {
            'vid_feat_dir': './data/anet_1.3/c3d_features',
            'anno_file': './data/anet_1.3/annotations',
            'max_text_len': 48,
            'max_vid_len': 1536,
            'clip_size': 16,
            'clip_stride': 8,
            'downsample_rate': None,
            'normalize': False,
            'to_fixed_len': False,
            'resize_scale': None,
            'jitter': None,
            'trunc_thresh': 0.5,
            'crop_ratio': (0.9, 1.0),
        },
    },

    'model': {
        'vid_dim': 500,
        'text_dim': 300,

        'common': {
            'embd_dim': 256,
            'n_heads': 4,
            'attn_pdrop': 0.1,
            'proj_pdrop': 0.1,
            'path_pdrop': 0.1,
        },

        'text_net': {
            'latent_size': 32,
            'n_layers': 5,
            'pe_type': 0,
        },

        'vid_net': {
            'arch': (2, 5),
            'mha_win_size': 9,
            'pe_type': 0,
        },

        'head': {
            'n_layers': 2,
            'mha_win_size': 5,
            'ada_type': 'adaattn',
            'norm_type': 'in',
            'pe_type': 0,
        },

        'pt_gen': {
            'regression_range': (
                (0, 8), 
                (4, 16), 
                (8, 32), 
                (16, 64), 
                (32, 128), 
                (64, 10000),
            ),
            'use_offset': False,
        },
        'neck': {
            'n_layers': 2,
            'mha_win_size': 5,
            'ada_type': 'adaattn',
            'norm_type': 'in',
            'pe_type': 0,
        },
    },

    'opt': {
        'epochs': 10,
        'warmup_epochs': 5,
        'ema_momentum': 0.999,
        # optimizer
        'optim_type': 'adamw',  # sgd | adam | adamw
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 0.05,
        # scheduler
        'sched_type': 'cosine', # cosine | multistep
        'steps': [],
        'gamma': 0.1,
    },

    'train': {
        'center_sampling': 'radius',
        'center_sampling_radius': 3,

        'loss_norm': 150,
        'loss_norm_momentum': 0.9,

        'cls_weight': 1.0,
        'reg_weight': 1.0,
        'iou_weight': 0.0,

        'label_smoothing': 0.2,
        'focal_alpha': 0.5,
        'focal_gamma': 2.0,

        'reg_gamma': None,
        'reg_log_scale': False,

        'iou_beta': 0.1,
    },

    'eval': {
        'use_iou_score': False,

        # pre-NMS filtering
        'pre_nms_thresh': 0.001,
        'pre_nms_topk': 2000,
        'seg_len_thresh': 0.1,

        # NMS
        'nms_mode': 'soft_nms',
        'iou_thresh': 0.1,
        'min_score': 0.01,
        'max_num_segs': 100,
        'sigma': 0.9,
        'voting_thresh': 0.95,
    },
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def _check_and_update_model_params(config):
    t_cfg = config['model']['text_net']
    v_cfg = config['model']['vid_net']
    h_cfg = config['model']['head']
    n_cfg = config['model']['neck']

    t_cfg['in_dim'] = config['model']['text_dim']
    v_cfg['in_dim'] = config['model']['vid_dim']

    max_text_len = config['data']['dataset']['max_text_len']
    max_vid_len = config['data']['dataset']['max_vid_len']
    t_cfg['max_seq_len'] = max_text_len
    v_cfg['max_seq_len'] = max_vid_len
    h_cfg['max_seq_len'] = max_vid_len
    n_cfg['max_seq_len'] = max_vid_len

    n_fpn_levels = v_cfg['arch'][-1] + 1
    h_cfg['n_fpn_levels'] = n_fpn_levels
    n_cfg['n_fpn_levels'] = n_fpn_levels

    t_cfg.update(config['model']['common'])
    v_cfg.update(config['model']['common'])
    h_cfg.update(config['model']['common'])
    n_cfg.update(config['model']['common'])


    # derive point generator parameters
    ## NOTE: buffer more points for longer sequence at inference time
    config['model']['pt_gen']['max_seq_len'] = max_vid_len * 4
    config['model']['pt_gen']['n_fpn_levels'] = n_fpn_levels
    assert len(config['model']['pt_gen']['regression_range']) == n_fpn_levels

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    
    _merge(defaults, config)
    _check_and_update_model_params(config)
    
    return config