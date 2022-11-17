import torch
import torch.nn as nn

from .backbones import make_backbone
from .heads import make_head
from .neck import make_neck


class PtTransformer(nn.Module):
    """
    Transformer based model for single-stage sentence grounding
    """
    def __init__(
        self,
        vid_net_cfg,    # video encoder config
        text_net_cfg,   # text encoder config
        neck_cfg,       # fusion neck config
        head_cfg,       # head config
    ):
        super(PtTransformer, self).__init__()

        self.vid_net = make_backbone('video_transformer', **vid_net_cfg)
        self.text_net = make_backbone('text_transformer', **text_net_cfg)
        self.neck_net = make_neck('fusion_net',**neck_cfg)
        self.cls_head = make_head('cls', **head_cfg)
        self.reg_head = make_head('reg', **head_cfg)

    def forward(self, vid, vid_mask, text, text_mask):
        """
        Args:
            vid (float tensor, (bs, c1, t1)): video feature sequences.
            vid_mask (bool tensor, (bs, 1, t1)): video mask.
            text (float tensor, (bs, c2, t2)): text feature sequences.
            text_mask (bool tensor, (bs, 1, t2)): text mask.


        Returns:
            fpn_logits (float tensor [l * (bs, p)]): logits from all levels.
            fpn_offsets (float tensor [l * (bs, p, 2)]): offsets from all levels.
            fpn_ious (float tensor [l * (bs, p)]): IoUs from all levels.
            fpn_masks (bool tensor [l * (bs, p)]): masks from all levels.
        """
        # video encoder
        fpn_vid, fpn_vid_masks = self.vid_net(vid, vid_mask)

        # text encoder
        text, text_mask = self.text_net(text, text_mask)

        # fusion network
        fpn_vid,fpn_vid_masks = self.neck_net(fpn_vid,fpn_vid_masks,text,text_mask)

        # heads
        fpn_logits, _ = self.cls_head(fpn_vid, fpn_vid_masks, text, text_mask)
        fpn_offsets, fpn_ious, fpn_masks = \
            self.reg_head(fpn_vid, fpn_vid_masks, text, text_mask)

        return fpn_logits, fpn_offsets, fpn_ious, fpn_masks,fpn_vid,text


class BufferList(nn.Module):

    def __init__(self, buffers):
        super(BufferList, self).__init__()

        for i, buf in enumerate(buffers):
            self.register_buffer(str(i), buf, persistent=False)

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class PtGenerator(nn.Module):
    """
    A generator for candidate points from specified FPN levels.
    """
    def __init__(
        self,
        max_seq_len,        # max sequence length
        n_fpn_levels,       # number of FPN levels
        regression_range,   # regression range for all specified levels
        use_offset=False,   # if True, align points at the middle of two tics
    ):
        super(PtGenerator, self).__init__()

        assert len(regression_range) == n_fpn_levels
        assert max_seq_len % 2 ** (n_fpn_levels - 1) == 0, (
            'max sequence length must be divisible by cumulative scale change'
        )

        self.max_seq_len = max_seq_len
        self.n_fpn_levels = n_fpn_levels
        self.regression_range = regression_range
        self.use_offset = use_offset

        # generate and buffer all candidate points
        self.buffer_points = self._generate_points()

    def _generate_points(self):
        # tics on the input grid
        tics = torch.arange(0, self.max_seq_len, 1.0)

        points_list = []
        for l in range(self.n_fpn_levels):
            stride = 2 ** l
            points = tics[::stride][:, None]                    # (t, 1)
            if self.use_offset:
                points += 0.5 * stride

            reg_range = torch.as_tensor(
                self.regression_range[l], dtype=torch.float
            )[None].repeat(len(points), 1)                      # (t, 2)
            stride = torch.as_tensor(
                stride, dtype=torch.float
            )[None].repeat(len(points), 1)                      # (t, 1)
            points = torch.cat([points, reg_range, stride], 1)  # (t, 4)
            points_list.append(points)

        return BufferList(points_list)

    def forward(self, fpn_n_points):
        """
        Args:
            fpn_n_points (int list [l]): number of points at specified levels.

        Returns:
            fpn_point (float tensor [l * (p, 4)]): candidate points from speficied levels.
        """
        assert len(fpn_n_points) == self.n_fpn_levels

        fpn_points = []
        for n_pts, pts in zip(fpn_n_points, self.buffer_points):
            assert n_pts <= len(pts), (
                'number of requested points {:d} cannot exceed max number '
                'of buffered points {:d}'.format(n_pts, len(pts))
            )
            fpn_points.append(pts[:n_pts])
        return fpn_points
