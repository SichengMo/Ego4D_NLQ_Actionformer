import torch
import torch.nn.functional as F
import numpy as np


def sigmoid_focal_loss(
        logits,
        target,
        alpha=None,
        gamma=2.0,
        mask=None,
        reduction='none',
):
    """
    Focal loss for binary classification.

    Args:
        logits (float tensor, (bs, ...)): predicted logits.
        target (float tensor, (bs, ...)): classification labels.
        alpha (float): weighting factor in range (0, 1) to balance positive
            vs. negative samples.
        gamma (float): focusing factor to balance easy vs. hard samples.
        mask (bool tensor, (...)): weight mask.
        reduction (str, 'none' | 'mean' | 'sum'):
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'sum': The output will be summed.

    Returns:
        Loss tensor with the reduction option applied.
    """
    # focal loss: FL(p_t) = -(1 - p_t) ** gamma * log(p_t)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    if gamma is not None:
        p_t = torch.exp(-loss)
        loss = loss * (1 - p_t) ** gamma

    if alpha is not None:
        assert alpha >= 0 and alpha < 1
        alpha_t = alpha * (target > 0.5) + (1 - alpha) * (target < 0.5)
        loss = alpha_t * loss

    if mask is not None:
        loss = loss * mask

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


def ctr_iou(x, y):
    lx, rx = x[..., 0], x[..., 1]
    ly, ry = y[..., 0], y[..., 1]

    # intersection key points
    lkis = torch.minimum(lx, ly)
    rkis = torch.minimum(rx, ry)

    # IOU
    intsct = rkis + lkis
    union = (lx + rx) + (ly + ry) - intsct
    iou = intsct / union.clamp(min=1e-8)

    return iou


def ctr_iou_loss(
        pred,
        target,
        gamma=None,
        log_scale=False,
        mask=None,
        reduction='none',
):
    """
    IOU loss for 1D intervals.

    This implementation assumes a 1D event is represented using the same
    center point with different offsets, i.e.,

    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0,

    and is equivalent to the generalized IOU loss.

    Args:
        pred (float tensor, (..., 2)): predicted offsets.
        target (float tensor, (..., 2)): target offsets.
        log_scale (bool): if True, return loss in log scale.
        mask (bool tensor, (...)): weight mask.
        reduction (str, 'none' | 'mean' | 'sum'):
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'sum': The output will be summed.

    Returns:
        Loss tensor with the reduction option applied.
    """
    # pred = pred.nan_to_num()
    assert pred.shape == target.shape
    assert (pred >= 0).all(), 'predicted offsets must be non-negative'
    assert (target >= 0).all(), 'target offsets must be non-negative'

    iou = ctr_iou(pred, target)
    if log_scale:
        loss = -torch.log(iou + 1e-8)
    else:
        loss = 1 - iou

    if gamma is not None:
        loss = loss ** gamma

    if mask is not None:
        loss = loss * mask

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()

    return loss


def ctr_diou_loss_1d(
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = 'none',
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-IoU Loss (Zheng et. al)
    https://arxiv.org/abs/1911.08287
    This is an implementation that assumes a 1D event is represented using
    the same center point with different offsets, e.g.,
    (t1, t2) = (c - o_1, c + o_2) with o_i >= 0
    Reference code from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/giou_loss.py
    Args:
        input/target_offsets (Tensor): 1D offsets of size (N, 2)
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """
    input_offsets = pred
    target_offsets = target
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()
    # check all 1D events are valid
    if not (input_offsets >= 0.0).all():
        print(list(input_offsets))
    assert (input_offsets >= 0.0).all(), "predicted offsets must be non-negative"
    assert (target_offsets >= 0.0).all(), "GT offsets must be non-negative"

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def smooth_l1_loss(
        pred,
        target,
        beta=0.1,
        reduction='none',
):
    return F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)


def check_tensor(x):
    if (True in x.isnan()) or (True in x.isinf()):
        print(True in x.isnan())
        print(True in x.isposinf())
        print(True in x.isneginf())
        return True
    else:
        return False


def nce_helper(v, T, t):
    return torch.sum(v @ T) / (torch.linalg.matrix_norm(T, dim=(0, 1)) * t)


def nce_loss_cal(
        gt_labels,
        vid_feat,
        text_feat,
        neg_feat,
        t,
):
    """
    Contrastive loss which use (correct frames, true text) as positive pairs and use
    (correct frames, false text) as negative pairs.

    https://arxiv.org/abs/2207.00383

    Args:
        gt_labels (float tensor, (bs, ...)): classification labels.
        vid_feat  (float tensor, (bs, ..., c1)): concatenated fpn video features.
        text_feat (float tensor, (bs, t2, c1)): input (correct) text features.
        neg_feat  (float tensor, (n*bs,t2, c1)): non-related (false) text features.
        t (float): temperature weighting factor.

    Returns:
        Loss tensor with sum reduction applied.
    """

    # normalize vid feature
    vid_feat = F.normalize(vid_feat, dim=2)

    gt_false = (~(gt_labels))

    # get length of positive video features
    gt_postive_sum = torch.sum(gt_labels.long(), dim=1).float()
    gt_postive_sum += torch.ones(gt_postive_sum.shape).cuda() * 1e-12

    # get score of (false vid frame - correct text ) pair
    vid_mask = gt_labels.unsqueeze(1).repeat(1, text_feat.shape[1], 1)
    nce_false = vid_feat.masked_fill(vid_mask, 0).view(vid_mask.shape)
    nce_false = nce_false.permute(0, 2, 1) @ (text_feat)

    text_norm = torch.linalg.matrix_norm(text_feat, dim=(1, 2)) * t
    text_norm = text_norm.unsqueeze(-1).unsqueeze(-1)
    text_norm += torch.ones(text_norm.shape).cuda() * 1e-12
    nce_false = torch.div(nce_false, text_norm)

    # original use torch.exp, but it leads to positive inf problem
    nce_false = torch.exp(nce_false.sum(dim=2)).sum(dim=1)

    # get score of (correct vid frame - true text ) pair
    vid_mask = gt_false.unsqueeze(1).repeat(1, text_feat.shape[1], 1)
    nce_true = vid_feat.masked_fill(vid_mask, 0).view(vid_mask.shape)
    nce_true = nce_true.permute(0, 2, 1) @ (text_feat)

    text_norm = torch.linalg.matrix_norm(text_feat, dim=(1, 2)) * t
    text_norm = text_norm.unsqueeze(-1).unsqueeze(-1)
    text_norm += torch.ones(text_norm.shape).cuda() * 1e-12

    nce_true = torch.div(nce_true, text_norm)
    nce_false_all = nce_false.unsqueeze(-1).repeat(1, nce_true.shape[1])
    nce_true = torch.sum(nce_true, dim=2)

    # original use torch.exp, but it leads to posinf problem
    nce_true = torch.exp(nce_true)

    # get nce score for each video
    nce_base = nce_true + nce_false_all
    nce_base_smooth = torch.ones(nce_base.shape).cuda() * 1e-12
    nce_base = nce_base + nce_base_smooth
    nce_score = torch.div(nce_true, nce_base)

    nce_score = nce_score.masked_fill(gt_false, 1).view(nce_score.shape)
    nce_score = -torch.log(nce_score)
    nce_score = nce_score.sum(dim=1)
    nce_score = torch.div(nce_score, gt_postive_sum)

    return torch.sum(nce_score)


def nce_loss_cal_all(
        gt_labels,
        vid_feat,
        text_feat,
        neg_feat,
        t,
        weight_pair1 = 0.5,
        weight_pair2 = 1,
):
    """
    Contrastive loss which use (correct frames, true text) as positive pairs and use
    (correct frames, false text) as negative pairs.

    https://arxiv.org/abs/2207.00383

    Args:
        gt_labels (float tensor, (bs, ...)): classification labels.
        vid_feat  (float tensor, (bs, ..., c1)): concatenated fpn video features.
        text_feat (float tensor, (bs, t2, c1)): input (correct) text features.
        neg_feat  (float tensor, (n*bs,t2, c1)): non-related (false) text features.
        t (float): temperature weighting factor.

    Returns:
        Loss tensor with sum reduction applied.
    """

    # normalize vid feature
    vid_feat = F.normalize(vid_feat, dim=2)

    # concatenate all negative text feature and repeat for bs times
    neg_feat = neg_feat.view(1, neg_feat.shape[1], neg_feat.shape[0] * neg_feat.shape[2])
    neg_feat = neg_feat.repeat(text_feat.shape[0], 1, 1)

    gt_false = (~(gt_labels))

    # get length of positive video features
    gt_postive_sum = torch.sum(gt_labels.long(), dim=1).float()
    gt_postive_sum += torch.ones(gt_postive_sum.shape).cuda() * 1e-12

    # get score of (false vid frame - correct text ) pair as negative pair 1
    vid_mask = gt_labels.unsqueeze(1).repeat(1, text_feat.shape[1], 1)
    nce_false = vid_feat.masked_fill(vid_mask, 0).view(vid_mask.shape)
    nce_false = nce_false.permute(0, 2, 1) @ (text_feat)

    text_norm = torch.linalg.matrix_norm(text_feat, dim=(1, 2)) * t
    text_norm = text_norm.unsqueeze(-1).unsqueeze(-1)
    text_norm += torch.ones(text_norm.shape).cuda() * 1e-12
    nce_false = torch.div(nce_false, text_norm)

    # original use torch.exp, but it leads to positive inf problem
    nce_false1 = torch.exp(nce_false.sum(dim=2)).sum(dim=1)

    # get score of (correct vid frame - fasle text ) pair as negative pair 2
    vid_mask = gt_labels.unsqueeze(1).repeat(1, text_feat.shape[1], 1)
    nce_false = vid_feat.masked_fill(vid_mask, 0).view(vid_mask.shape)
    nce_false = nce_false.permute(0, 2, 1) @ (neg_feat)

    text_norm = torch.linalg.matrix_norm(neg_feat, dim=(1, 2)) * t
    text_norm = text_norm.unsqueeze(-1).unsqueeze(-1)
    text_norm += torch.ones(text_norm.shape).cuda() * 1e-12
    nce_false = torch.div(nce_false, text_norm)

    # original use torch.exp, but it leads to positive inf problem
    nce_false2 = torch.exp(nce_false.sum(dim=2)).sum(dim=1)

    # sum up false pair score
    nce_false = weight_pair1 * nce_false1 + weight_pair2 * nce_false2

    # get score of (correct vid frame - true text ) pair
    vid_mask = gt_false.unsqueeze(1).repeat(1, text_feat.shape[1], 1)
    nce_true = vid_feat.masked_fill(vid_mask, 0).view(vid_mask.shape)
    nce_true = nce_true.permute(0, 2, 1) @ (text_feat)

    text_norm = torch.linalg.matrix_norm(text_feat, dim=(1, 2)) * t
    text_norm = text_norm.unsqueeze(-1).unsqueeze(-1)
    text_norm += torch.ones(text_norm.shape).cuda() * 1e-12

    nce_true = torch.div(nce_true, text_norm)
    nce_false_all = nce_false.unsqueeze(-1).repeat(1, nce_true.shape[1])
    nce_true = torch.sum(nce_true, dim=2)

    # original use torch.exp, but it leads to posinf problem
    nce_true = torch.exp(nce_true)

    # get nce score for each video
    nce_base = nce_true + nce_false_all
    nce_base_smooth = torch.ones(nce_base.shape).cuda() * 1e-12
    nce_base = nce_base + nce_base_smooth
    nce_score = torch.div(nce_true, nce_base)

    nce_score = nce_score.masked_fill(gt_false, 1).view(nce_score.shape)
    nce_score = -torch.log(nce_score)
    nce_score = nce_score.sum(dim=1)
    nce_score = torch.div(nce_score, gt_postive_sum)

    return torch.sum(nce_score)
