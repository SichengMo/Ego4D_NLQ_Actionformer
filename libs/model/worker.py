from copy import deepcopy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import PtTransformer, PtGenerator
from .losses import *
from ..utils import batched_nms


class Worker():
    """
    A worker for TSG training and inference
    """

    def __init__(self, model_cfg):

        self.model_cfg = model_cfg
        self.parallel = False

        # unroll model specs
        vid_net_cfg = model_cfg['vid_net']
        text_net_cfg = model_cfg['text_net']
        neck_net_cfg = model_cfg['neck']
        head_cfg = model_cfg['head']
        pt_gen_cfg = model_cfg['pt_gen']

        # build model
        self.model = PtTransformer(
            vid_net_cfg=vid_net_cfg,
            text_net_cfg=text_net_cfg,
            neck_cfg=neck_net_cfg,
            head_cfg=head_cfg,
        )
        self.ema_model = deepcopy(self.model)
        self.pt_gen = PtGenerator(**pt_gen_cfg)

        # compute derived fields
        n_fpn_levels = pt_gen_cfg['n_fpn_levels']
        mha_win_size = vid_net_cfg['mha_win_size']
        if isinstance(mha_win_size, int):
            mha_win_size = (mha_win_size,) * n_fpn_levels

        ds_strides = [2 ** i for i in range(n_fpn_levels)]
        min_chunk_size = 1
        # ensures at least one chunk is available at the highest FPN level
        for idx in range(len(ds_strides)):
            stride = ds_strides[idx]
            if mha_win_size[idx] > 0:
                stride *= (mha_win_size[idx] // 2) * 2
            min_chunk_size = max(min_chunk_size, stride)

        self.max_vid_len = vid_net_cfg['max_seq_len']
        self.max_text_len = text_net_cfg['max_seq_len']
        assert self.max_vid_len % min_chunk_size == 0, (
            'max video length must be a multiple of {:d}'.format(min_chunk_size)
        )
        self.min_chunk_size = min_chunk_size

    def cuda(self, parallel=False):
        self.model.cuda()
        self.ema_model.cuda()
        self.pt_gen.cuda()
        if parallel:
            self.model = nn.DataParallel(self.model)
            self.parallel = True

    def load(self, ckpt):
        if self.parallel:
            self.model.module.load_state_dict(ckpt['model'])
        else:
            self.model.load_state_dict(ckpt['model'])
        self.ema_model.load_state_dict(ckpt['ema_model'])

    def save(self):
        if self.parallel:
            model = self.model.module.state_dict()
        else:
            model = self.model.state_dict()
        ema_model = self.ema_model.state_dict()
        ckpt = {'model': model, 'ema_model': ema_model}
        return ckpt

    def state_dict(self):
        if self.parallel:
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()

    def parameters(self):
        if self.parallel:
            return self.model.module.parameters()
        else:
            return self.model.parameters()

    def named_parameters(self):
        if self.parallel:
            return self.model.module.named_parameters()
        else:
            return self.model.named_parameters()

    def named_modules(self):
        if self.parallel:
            return self.model.module.named_modules()
        else:
            return self.model.named_modules()

    def ema_update(self, momentum=0.999):
        state_dict = self.state_dict()
        ema_state_dict = self.ema_model.state_dict()
        for ema_v, v in zip(ema_state_dict.values(), state_dict.values()):
            ema_v.copy_(momentum * ema_v + (1.0 - momentum) * v)

    @torch.no_grad()
    def _batchify(self, vid_list, text_list):
        """
        Put vid and text features and their masks in a batch.

        Args:
            vid_list (float tensor [bs * (c1, t1)]): a list of vid features.
            text_list (float tensor [bs * (c2, t2)]): a list of text features.
            neg_text_list (float tensor [bs * [n * (c2, t1)]]): a list of negative text features.

        Returns:
            vid (float tensor, (bs, c1, t1)): vid feature sequences.
            vid_masks (bool tensor, (bs, 1, t1)): vid masks.
            text (float tensor, (bs, c2, t2)): text feature sequences.
            text_masks (bool tensor, (bs, 1, t2)): text masks.
            neg_text (float tensor, (bs*n, c2, t2)): negative text feature sequences.
            neg_text_masks (bool tensor, (bs*n, 1, t2)): negative text masks.
        """
        assert len(vid_list) == len(text_list)
        bs = len(vid_list)

        # video features
        vid_dim = vid_list[0].size(0)
        vid_lens = [v.size(-1) for v in vid_list]
        max_vid_len = max(vid_lens)

        # text features
        text_dim = text_list[0].size(0)
        text_lens = [t.size(-1) for t in text_list]
        max_text_len = max(text_lens)



        # batch video and text features
        assert max_vid_len <= self.max_vid_len
        assert max_text_len <= self.max_text_len

        vid = torch.zeros(bs, vid_dim, self.max_vid_len)
        text = torch.zeros(bs, text_dim, self.max_text_len)

        for idx in range(bs):
            vid[idx, :, :vid_lens[idx]].copy_(vid_list[idx])
            text[idx, :, :text_lens[idx]].copy_(text_list[idx])

        # batch video and text masks
        vid_lens = torch.as_tensor(vid_lens).view(bs, 1, 1)
        text_lens = torch.as_tensor(text_lens).view(bs, 1, 1)
        vid_masks = torch.arange(vid.size(-1)).view(1, 1, -1) < vid_lens
        text_masks = torch.arange(text.size(-1)).view(1, 1, -1) < text_lens


        return vid, vid_masks, text, text_masks

    @torch.no_grad()
    def _annotate_points(self, points, seg_list, **kwargs):
        """
        Assign ground-truth labels and offsets to candidate points.

        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride(1))
            seg_list (float [bs * [2]]): ground-truth segments.

        Returns:
            labels (bool tensor, (bs, p)): ground-truth binary labels.
            offsets (float tensor, (bs, p, 2)): ground-truth offsets.
        """
        labels_list, offsets_list = [], []

        for seg in seg_list:
            labels, offsets = \
                self._annotate_points_per_video(points, seg, **kwargs)
            labels_list.append(labels)
            offsets_list.append(offsets)

        labels = torch.stack(labels_list)
        offsets = torch.stack(offsets_list)

        return labels, offsets

    @torch.no_grad()
    def _annotate_points_per_video(self, points, seg, **kwargs):
        """
        Args:
            points (float tensor, (p, 4)): candidate points from all levels.
                (coordinate (1), regression range (2), stride (1))
            seg (float [2]): ground-truth segment.

        Returns:
            labels (bool tensor, (p,)): ground-truth binary labels.
            offsets (float tensor, (p, 2)): ground-truth offsets.
        """
        # point distance to segment boundaries
        pt2start = points[:, 0] - seg[0]  # (p,)
        pt2end = seg[1] - points[:, 0]  # (p,)

        # offsets rescaled by down-sampling stride
        offsets = torch.stack([pt2start, pt2end], dim=-1) / points[:, 3:]

        # (1) whether a point lies in given sampling window
        if kwargs['center_sampling'] == 'radius':
            ctr = 0.5 * (seg[0] + seg[1])
            radius = points[:, 3] * kwargs['center_sampling_radius']
            t_min = (ctr - radius).clamp(min=seg[0])
            t_max = (ctr + radius).clamp(max=seg[1])
            # point distance to window boundaries
            pt2left = points[:, 0] - t_min  # (p,)
            pt2right = t_max - points[:, 0]  # (p,)
            inside_window = torch.logical_and(pt2left > 0, pt2right > 0)
        else:
            inside_window = torch.logical_and(pt2start > 0, pt2end > 0)

        # (2) whether event is within regression range of a point
        max_reg_dist = torch.maximum(pt2start, pt2end)
        inside_range = torch.logical_and(
            max_reg_dist >= points[:, 1], max_reg_dist < points[:, 2]
        )

        # a point is positive only if it meets both criteria
        labels = torch.logical_and(inside_window, inside_range)

        return labels, offsets

    def _calculate_loss(
            self,
            pred_logits,
            pred_offsets,
            pred_ious,
            gt_labels,
            gt_offsets,
            masks,
            vid_feats,
            text_feats,
            neg_feats,
            **kwargs
    ):
        """
        Args:
            pred_logits (float tensor, (bs, p)): predicted logits.
            pred_offsets (float tensor, (bs, p, 2)): predicted offsets.
            pred_ious (float tensor, (bs, p)): predicted IoUs.
            gt_labels (bool tensor, (bs, p)): ground-truth labels.
            gt_offsets (float tensor, (bs, p, 2)): ground-truth offsets.
            masks (bool tensor, (bs, p)): mask (1 for valid points).
        """
        loss_dict = dict()
        pos_masks = torch.logical_and(gt_labels, masks)

        vid_feats = torch.cat(vid_feats, dim=-1)  # (bs, p)

        # offsets = torch.cat(fpn_offsets, dim=1)     # (bs, p, 2)

        gt_labels_orig = gt_labels

        # momentum update of normalization factor
        loss_norm = pos_masks.sum().item()
        loss_norm = \
            kwargs['loss_norm_momentum'] * kwargs['loss_norm'] \
            + (1.0 - kwargs['loss_norm_momentum']) * max(loss_norm, 1)
        loss_dict['loss_norm'] = loss_norm

        cls_weight = kwargs.get('cls_weight', 1)
        reg_weight = kwargs.get('reg_weight', 1)
        iou_weight = kwargs.get('iou_weight', 0)
        nce_weight = kwargs.get('nce_weight', 1)
        # label smoothing as regularization
        gt_labels = gt_labels.float()
        gt_labels *= 1.0 - kwargs['label_smoothing']
        gt_labels += kwargs['label_smoothing'] / 2

        # classification loss (only defined on valid points)
        pred_logits = pred_logits[masks]
        gt_labels = gt_labels[masks]

        cls_loss = sigmoid_focal_loss(
            logits=pred_logits,
            target=gt_labels,
            alpha=kwargs.get('focal_alpha'),
            gamma=kwargs.get('focal_gamma'),
            reduction='sum',
        ) / loss_norm

        # regression loss (only defined on positive points)
        pred_offsets = pred_offsets[pos_masks]
        gt_offsets = gt_offsets[pos_masks]

        reg_loss = ctr_iou_loss(
            pred=pred_offsets,
            target=gt_offsets,
            gamma=kwargs.get('reg_gamma'),
            log_scale=kwargs.get('reg_log_scale'),
            reduction='sum',
        ) / loss_norm

        # uncomment the following lines to switch to dIoU loss

        # reg_loss = ctr_diou_loss_1d(
        #     pred=pred_offsets,
        #     target=gt_offsets,
        #     reduction='sum',
        # ) / loss_norm

        loss_dict['cls'] = cls_loss
        loss_dict['reg'] = reg_loss
        total_loss = cls_weight * cls_loss + reg_weight * reg_loss

        # IoU loss (only defined on positive points)
        if iou_weight > 0:
            pred_ious = torch.sigmoid(pred_ious[pos_masks])
            gt_ious = ctr_iou(pred_offsets, gt_offsets).detach()

            iou_loss = smooth_l1_loss(
                pred=pred_ious,
                target=gt_ious,
                beta=kwargs.get('iou_beta', 0.1),
                reduction='sum',
            ) / loss_norm

            loss_dict['iou'] = iou_loss
            total_loss += iou_weight * iou_loss

        # regression loss (only defined on valid points)
        if nce_weight > 0:
            nce_loss = nce_loss_cal(
                gt_labels_orig,
                vid_feats,
                text_feats,
                None,
                t=kwargs.get('nce_temp', 0.07)
            ) / loss_norm
            loss_dict['nce'] = nce_loss

            total_loss += nce_weight * nce_loss

        loss_dict['total'] = total_loss
        return loss_dict

    @torch.no_grad()
    def _collect_segments(
            self,
            fpn_points,
            fpn_logits,
            fpn_offsets,
            fpn_ious,
            fpn_masks,
            **kwargs
    ):
        """
        Args:
            fpn_points (float tensor [l * (p, 4)]): candidate points from all levels.
            fpn_logits (float tensor [l * (1, p)]): predicted logits from all levels.
            fpn_offsets (float tensor [l * (1, p, 2)]): predicted offsets from all levels.
            fpn_ious (float tensor, [l * (1, p)]): predicted IoUs from all levels.
            fpn_masks (bool tensor [l * (1, p)]): masks for all levels.

        Returns:
            segs (float tensor, (n, 2)): candidate segments.
            scores (float tensor, (n,)): segment scores.
        """
        fpn_segs, fpn_scores = [], []
        max_score = 0

        # loop over all FPN levels
        for points, logits, offsets, ious, masks in zip(
                fpn_points, fpn_logits, fpn_offsets, fpn_ious, fpn_masks
        ):
            # batch size defaults to 1 at test time
            # points: (p, 4), masks: (p,), logits: (p,), offsets: (p, 2), ious: (p,)
            assert len(logits) == 1;
            logits = logits[0]
            assert len(offsets) == 1;
            offsets = offsets[0]
            assert len(ious) == 1;
            ious = ious[0]
            assert len(masks) == 1;
            masks = masks[0]

            # compute confidence scores
            scores = torch.sigmoid(logits)
            if kwargs.get('use_iou_score'):
                scores *= torch.sigmoid(ious)
            scores *= masks.float()

            # clean up predictions before NMS for efficiency
            ## (1) filter points by confidence threshold
            idx = scores > kwargs['pre_nms_thresh']
            points, scores, offsets = points[idx], scores[idx], offsets[idx]

            ## (2) only keep top-k scoring boxes
            n_topk = min(len(idx), kwargs['pre_nms_topk'])
            idx = scores.argsort(descending=True)[:n_topk]
            points, scores, offsets = points[idx], scores[idx], offsets[idx]

            ## (3) compute predicted segments
            left = points[:, 0] - offsets[:, 0] * points[:, 3]
            right = points[:, 0] + offsets[:, 1] * points[:, 3]
            segs = torch.stack([left, right], dim=-1)

            ## (4) filter segments by length threshold
            seg_lens = right - left
            idx = seg_lens > kwargs['seg_len_thresh']
            segs, scores = segs[idx], scores[idx]

            fpn_segs.append(segs)
            fpn_scores.append(scores)

        segs = torch.cat(fpn_segs)
        scores = torch.cat(fpn_scores)

        return segs, scores

    def train(self, data_list, cfg):
        # batch data
        vid_list = [d['vid_feats'] for d in data_list]
        text_list = [d['text_feats'] for d in data_list]
        vid, vid_masks, text, text_masks = self._batchify(vid_list, text_list)

        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)


        # forward pass
        self.model.train()
        fpn_logits, fpn_offsets, fpn_ious, fpn_masks, fpn_vid, text = \
            self.model(vid, vid_masks, text, text_masks)
        fpn_n_points = [f.size(1) for f in fpn_logits]
        fpn_points = self.pt_gen(fpn_n_points)

        # stitch model outputs
        logits = torch.cat(fpn_logits, dim=1)  # (bs, p)
        offsets = torch.cat(fpn_offsets, dim=1)  # (bs, p, 2)
        ious = torch.cat(fpn_ious, dim=1)  # (bs, p)
        masks = torch.cat(fpn_masks, dim=1)  # (bs, p)
        points = torch.cat(fpn_points)  # (p, 4)

        # annotate points
        seg_list = [d['target'] for d in data_list]
        gt_labels, gt_offsets = \
            self._annotate_points(points, seg_list, **cfg)

        # calculate loss
        loss_dict = self._calculate_loss(
            logits, offsets, ious, gt_labels, gt_offsets, masks, fpn_vid, text, None, **cfg
        )
        cfg['loss_norm'] = loss_dict['loss_norm']

        return loss_dict

    @torch.no_grad()
    def eval(self, data, cfg, ema=True):
        # prepare data (batch size = 1)
        vid, text = data['vid_feats'], data['text_feats']
        vid_len, max_vid_len = vid.size(-1), self.max_vid_len
        if vid_len > max_vid_len:
            # pad vid features to the next divisible size
            ## NOTE: this ensures the sequence can be perfectly chunked
            ## for efficient local attention
            stride = self.min_chunk_size
            max_vid_len = (vid_len + (stride - 1)) // stride * stride
        vid = F.pad(vid, (0, max_vid_len - vid_len))

        vid, text = vid[None], text[None]
        vid_masks = torch.arange(max_vid_len).view(1, 1, -1) < vid_len
        text_masks = torch.ones((1, 1, text.size(-1)), dtype=torch.bool)

        vid = vid.cuda(non_blocking=True)
        vid_masks = vid_masks.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        text_masks = text_masks.cuda(non_blocking=True)

        # forward pass
        model = self.ema_model if ema else self.model
        model.eval()
        fpn_logits, fpn_offsets, fpn_ious, fpn_masks, _, _ = \
            model(vid, vid_masks, text, text_masks)
        fpn_n_points = [f.size(1) for f in fpn_logits]
        fpn_points = self.pt_gen(fpn_n_points)

        # collect segments and their scores
        segs, scores = self._collect_segments(
            fpn_points, fpn_logits, fpn_offsets, fpn_ious, fpn_masks, **cfg
        )

        # NMS
        segs, scores = batched_nms(
            segs.cpu(), scores.cpu(),
            iou_thresh=cfg['iou_thresh'],
            min_score=cfg['min_score'],
            max_num_segs=cfg['max_num_segs'],
            mode=cfg['nms_mode'],
            sigma=cfg['sigma'],
            voting_thresh=cfg['voting_thresh'],
        )

        # convert segments to timestamps in seconds
        if len(segs) > 0:
            clip_stride = data['clip_stride']
            clip_size = data['clip_size']
            fps = data['fps']
            duration = data['duration']

            segs = (segs * clip_stride + 0.5 * clip_size) / fps
            segs = torch.clamp(segs, min=0, max=duration)

        results = {
            'id': data['id'],
            'segments': segs,
            'scores': scores,
        }

        return results
