import torch.nn as nn
import torch.nn.functional as F

from .blocks import (get_sinusoid_encoding, TransformerEncoderBlock, Scale)

heads = dict()


def register_head(name):
    def decorator(module):
        heads[name] = module
        return module

    return decorator


@register_head('cls')
class ClassificationHead(nn.Module):
    """
    1D Convolutional head for event classification
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            max_seq_len,  # max sequence length
            n_fpn_levels,  # number of FPN levels
            n_layers=2,  # number of transformer layers
            n_heads=4,  # number of attention heads for MHA
            mha_win_size=5,  # local window size for MHA
            attn_pdrop=0.1,  # dropout rate for attention maps
            proj_pdrop=0.1,  # dropout rate for projection MLPs
            path_pdrop=0.1,  # dropout rate for residual paths
            pe_type=0,  # position encoding type (-1, 0, 1)
            **kwargs,
    ):
        super(ClassificationHead, self).__init__()

        self.max_seq_len = max_seq_len
        self.n_fpn_levels = n_fpn_levels

        # position encoding (c, t)
        assert pe_type in (-1, 0, 1)
        self.pe_type = pe_type
        if pe_type >= 0:
            pe = get_sinusoid_encoding(max_seq_len, embd_dim // 2, pe_type)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # self-attention
        self.transformer = nn.ModuleList()
        for idx in range(n_layers):
            self.transformer.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=-1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    out_ln=False,
                )
            )

        # FFN
        self.heads = nn.ModuleList()
        if self.n_fpn_levels >= 3:
            for i in range(0, 3):
                self.heads.append(
                    nn.Sequential(
                        nn.Conv1d(embd_dim, embd_dim, 1), nn.ReLU(inplace=True),
                        nn.Conv1d(embd_dim, embd_dim, 1), nn.ReLU(inplace=True),
                        nn.Conv1d(embd_dim, 1, 1),
                    )
                )
        self.head = nn.Sequential(
            nn.Conv1d(embd_dim, embd_dim, 1), nn.ReLU(inplace=True),
            nn.Conv1d(embd_dim, embd_dim, 1), nn.ReLU(inplace=True),
            nn.Conv1d(embd_dim, 1, 1),
        )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, fpn_vid, fpn_vid_masks, text, text_mask):
        assert len(fpn_vid) == self.n_fpn_levels
        assert len(fpn_vid_masks) == self.n_fpn_levels

        bs, _, t = fpn_vid[0].size()
        out_logits, out_masks = tuple(), tuple()
        for l, (vid, vid_mask) in enumerate(zip(fpn_vid, fpn_vid_masks)):
            if l <= 1:
                logits = self.heads[2](vid) * vid_mask.float()  # (bs, 1, p)
            else:
                logits = self.heads[2](vid) * vid_mask.float()
            logits = logits.squeeze(1)  # (bs, p)
            vid_mask = vid_mask.squeeze(1)  # (bs, p)
            out_logits += (logits,)
            out_masks += (vid_mask,)

        return out_logits, out_masks

@register_head('reg')
class RegressionHead(nn.Module):
    """
    1D Convolutional head for offset regression
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            max_seq_len,  # max sequence length
            n_fpn_levels,  # number of FPN levels
            n_layers=2,  # number of transformer layers
            n_heads=4,  # number of attention heads for MHA
            mha_win_size=5,  # local window size for MHA
            attn_pdrop=0.1,  # dropout rate for attention maps
            proj_pdrop=0.1,  # dropout rate for projection MLPs
            path_pdrop=0.1,  # dropout rate for residual paths
            pe_type=0,  # position encoding type (-1, 0, 1)
            **kwargs,
    ):
        super(RegressionHead, self).__init__()

        self.max_seq_len = max_seq_len
        self.n_fpn_levels = n_fpn_levels

        # position encoding (c, t)
        assert pe_type in (-1, 0, 1)
        self.pe_type = pe_type
        if pe_type >= 0:
            pe = get_sinusoid_encoding(max_seq_len, embd_dim // 2, pe_type)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # self-attention
        self.transformer = nn.ModuleList()
        for idx in range(n_layers):
            self.transformer.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=-1,
                    n_heads=n_heads,
                    window_size=mha_win_size,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    out_ln=False,
                )
            )

        # FFN
        self.reg_heads = nn.ModuleList()
        if self.n_fpn_levels >= 3:
            for i in range(0, 3):
                self.reg_heads.append(
                    nn.Sequential(
                        nn.Conv1d(embd_dim, embd_dim, 1), nn.ReLU(inplace=True),
                        nn.Conv1d(embd_dim, embd_dim, 1), nn.ReLU(inplace=True),
                        nn.Conv1d(embd_dim, 2, 1),
                    )
                )

        self.iou_heads = nn.ModuleList()
        if self.n_fpn_levels >= 3:
            for i in range(0, 3):
                self.iou_heads.append(
                    nn.Sequential(
                        nn.Conv1d(embd_dim, embd_dim, 1), nn.ReLU(inplace=True),
                        nn.Conv1d(embd_dim, embd_dim, 1), nn.ReLU(inplace=True),
                        nn.Conv1d(embd_dim, 1, 1),
                    )
                )
        self.scales = nn.ModuleList([Scale() for _ in range(n_fpn_levels)])

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, fpn_vid, fpn_vid_masks, text, text_mask):
        assert len(fpn_vid) == self.n_fpn_levels
        assert len(fpn_vid_masks) == self.n_fpn_levels

        bs, _, t = fpn_vid[0].size()

        out_offsets, out_ious, out_masks = tuple(), tuple(), tuple()
        for l, (vid, vid_mask) in enumerate(zip(fpn_vid, fpn_vid_masks)):
            offsets = self.reg_heads[2](vid) * vid_mask.float()  # (bs, 2, p)
            offsets = F.relu(self.scales[2](offsets))
            ious = self.iou_heads[2](vid) * vid_mask.float()  # (bs, 1, p)
            offsets = offsets.transpose(1, 2)  # (bs, p, 2)
            ious = ious.squeeze(1)  # (bs, p)
            vid_mask = vid_mask.squeeze(1)  # (bs, p)

            out_offsets += (offsets,)
            out_ious += (ious,)
            out_masks += (vid_mask,)

        return out_offsets, out_ious, out_masks

def make_head(name, **kwargs):
    return heads[name](**kwargs)