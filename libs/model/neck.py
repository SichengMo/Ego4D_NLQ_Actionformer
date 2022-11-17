import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (get_sinusoid_encoding, TransformerEncoderBlock, MaskedAdaAttN, Scale, MaskedCrossAtten,MaskedConv1D)

necks = dict()


def register_neck(name):
    def decorator(module):
        necks[name] = module
        return module

    return decorator


@register_neck('fusion_net')
class FusionNeck(nn.Module):
    """
    1D Transformer network for feature fusion
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
            ada_type='adain',  # feature modulation method ('adain', 'adaattn')
            norm_type='in',  # feature normalization method ('in', 'ln')
            pe_type=0,  # position encoding type (-1, 0, 1)
            **kwargs,
    ):
        super(FusionNeck, self).__init__()

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
        self.ada = nn.ModuleList()
        self.Conv = nn.ModuleList()

        for idx in range(n_layers):

            # adding transformer block
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

            # add Conv Block
            self.Conv.append(
                MaskedConv1D(
                    embd_dim,
                    embd_dim,
                    kernel_size=3,
                    stride=1,
                    padding=3//2,
                )
            )

            if ada_type == 'adaattn':
                self.ada.append(
                    MaskedAdaAttN(
                        embd_dim,
                        n_heads=n_heads,
                        attn_pdrop=attn_pdrop,
                        norm_type=norm_type,
                    )
                )
            else:
                self.ada.append(
                    MaskedAdaAttN(
                        embd_dim,
                        n_heads=n_heads,
                        attn_pdrop=attn_pdrop,
                        norm_type=norm_type,
                    )
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

        # position encoding
        pe = self.pe
        if self.pe_type >= 0:
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
                pe = pe[..., :t]
            pe = pe * fpn_vid_masks[0].float()

        out_feats, out_masks = tuple(), tuple()
        for l, (vid, vid_mask) in enumerate(zip(fpn_vid, fpn_vid_masks)):
            if pe is not None and False:
                vid = vid + pe
                pe = pe[..., ::2]
            for idx in range(len(self.transformer)):

                vid, vid_mask = self.ada[idx](vid, vid_mask, text, text_mask)
                vid, _ = self.Conv[idx](vid, vid_mask)

            vid_feat = vid

            out_feats += (vid_feat,)
            out_masks += (vid_mask,)

        return out_feats, out_masks

def make_neck(name, **kwargs):
    return necks[name](**kwargs)