import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .blocks import (get_sinusoid_encoding,
                     TransformerEncoderBlock, TransformerDecoderBlock,
                     ConvBlock, MaskedConv1D, LayerNorm, TransformerEncoderBlockTopDown)

backbones = dict()


def register_backbone(name):
    def decorator(module):
        backbones[name] = module
        return module

    return decorator


@register_backbone('text_perceiver')
class TextPerceiver(nn.Module):
    """
    A perceiver encoder for text embedding.

    word embeddings
    -> [embedding projection]
    -> [xattn transformer]
    -> [self-attn transformer x L]
    -> latent word embeddings
    """

    def __init__(
            self,
            in_dim,  # input dimension
            embd_dim,  # embedding dimension
            latent_size,  # size of latent embeddings
            max_seq_len,  # max sequence length
            n_heads=4,  # number of attention heads
            n_layers=2,  # number of self-attention layers
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            pe_type=0,  # position encoding type (-1, 0, 1)
            **kwargs,
    ):
        super(TextPerceiver, self).__init__()

        self.max_seq_len = max_seq_len

        # embedding projection
        self.embd_fc = None
        if in_dim != embd_dim:
            self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # position encoding (c, t)
        assert pe_type in (-1, 0, 1)
        self.pe_type = pe_type
        if pe_type >= 0:
            pe = get_sinusoid_encoding(max_seq_len, embd_dim // 2, pe_type)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # fixed-size latent array
        self.latent_size = latent_size
        self.latents = nn.Embedding(latent_size, embd_dim)

        # embedding cross-attention
        self.embd_xattn = TransformerDecoderBlock(
            embd_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            use_self_attn=False,
        )

        # encoder
        self.transformer = nn.ModuleList()
        for _ in range(n_layers):
            self.transformer.append(
                TransformerEncoderBlock(
                    embd_dim,
                    n_heads=n_heads,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        bs, _, t = x.size()

        # embedding projection
        if self.embd_fc is not None:
            x, _ = self.embd_fc(x, mask)

        # position encoding
        if self.pe_type >= 0:
            pe = self.pe
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
                pe = pe[..., :t]
            x = x + pe * mask.float()

        # embedding cross-attention
        q = self.latents.weight.transpose(0, 1).expand(bs, -1, -1)  # (bs, d, n)
        q_mask = q.new_ones(bs, 1, self.latent_size).bool()  # (bs, 1, n)
        x, mask = self.embd_xattn(q, q_mask, x, mask)

        # encoder
        for idx in range(len(self.transformer)):
            x, _ = self.transformer[idx](x, mask)

        return x, mask


@register_backbone('text_transformer')
class TextTransformer(nn.Module):
    """
    A transformer encoder for text embedding.

    word embeddings
    -> [embedding projection]
    -> [self-attn transformer x L]
    -> latent word embeddings
    """

    def __init__(
            self,
            in_dim,  # text feature dimension
            embd_dim,  # embedding dimension
            max_seq_len,  # max sequence length
            n_heads=4,  # number of attention heads
            n_layers=5,  # number of transformer encoder layers
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            pe_type=0,  # position encoding type (-1, 0, 1)
            **kwargs,
    ):
        super(TextTransformer, self).__init__()

        self.max_seq_len = max_seq_len

        # embedding projection
        self.embd_fc = None
        if in_dim != embd_dim:
            self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # position encoding (c, t)
        assert pe_type in (-1, 0, 1)
        self.pe_type = pe_type
        if pe_type >= 0:
            pe = get_sinusoid_encoding(max_seq_len, embd_dim // 2, pe_type)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # self-attention transformers
        self.transformer = nn.ModuleList()
        for _ in range(n_layers):
            self.transformer.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=-1,
                    n_heads=n_heads,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        bs, _, t = x.size()

        # embedding projection
        if self.embd_fc is not None:
            x, _ = self.embd_fc(x, mask)

        # add position encoding
        if self.pe_type >= 0:
            pe = self.pe
            if self.training:
                assert t <= self.max_seq_len
            else:
                if t > self.max_seq_len:
                    pe = F.interpolate(
                        pe[None], size=t, mode='linear', align_corners=True
                    )[0]
                pe = pe[..., :t]
            x = x + pe * mask.float()

        # self-attention transformers
        for idx in range(len(self.transformer)):
            x, _ = self.transformer[idx](x, mask)

        return x, mask


@register_backbone('text_net')
class TextNet(nn.Module):

    def __init__(
            self,
            in_dim,  # text feature dimension
            embd_dim,  # embedding dimension
            max_seq_len,  # max sequence length
            n_heads=4,  # number of attention heads
            n_layers=5,  # number of transformer encoder layers
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            pe_type=0,  # position encoding type (-1, 0, 1)
            **kwargs,
    ):
        super(TextNet, self).__init__()

        self.max_seq_len = max_seq_len

        # embedding projection
        self.embd_fc = None
        if in_dim != embd_dim:
            self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)



        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        bs, _, t = x.size()

        # embedding projection
        if self.embd_fc is not None:
            x, _ = self.embd_fc(x, mask)

        return x, mask


@register_backbone('video_transformer')
class VideoTransformer(nn.Module):
    """
        A backbone that combines convolutions with transformer encoder layers
        to build a feature pyramid.
        [embedding convs] -> [stem transformers] -> [branch transformers]
    """

    def __init__(
            self,
            in_dim,  # video feature dimension
            embd_dim,  # embedding dimension
            max_seq_len,  # max sequence length
            arch=(2, 5),  # (#stem transformers, #branch transformers)
            n_heads=4,  # number of attention heads for MHA
            mha_win_size=-1,  # local window size for MHA
            attn_pdrop=0.1,  # dropout rate for attention maps
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            use_rel_pe=False,  # if True, use relative position encoding
            pe_type=0,  # position encoding type (-1, 0, 1)
            **kwargs,
    ):
        super(VideoTransformer, self).__init__()

        assert len(arch) == 2, '(#stem transformers, #branch transformers)'

        self.max_seq_len = max_seq_len

        if isinstance(mha_win_size, int):
            mha_win_size = (mha_win_size,) * (arch[-1] + 1)
        assert isinstance(mha_win_size, (list, tuple))
        assert len(mha_win_size) == arch[-1] + 1
        # position encoding (c, t)
        assert pe_type in (-1, 0, 1)
        self.pe_type = pe_type
        if pe_type >= 0:
            pe = get_sinusoid_encoding(max_seq_len, embd_dim // 2, pe_type)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # embedding projection
        self.embd_fc = None
        if in_dim != embd_dim:
            self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # stem transformer blocks
        self.stem = nn.ModuleList()
        for _ in range(arch[0]):
            self.stem.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=1,
                    n_heads=n_heads,
                    window_size=mha_win_size[0],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            )

        # branch transformer blocks
        self.branch = nn.ModuleList()
        for idx in range(arch[1]):
            self.branch.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=2,
                    n_heads=n_heads,
                    window_size=mha_win_size[idx + 1],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            )

        self.latlayers = nn.ModuleList()
        for idx in range(arch[1] + 1):
            self.latlayers.append(
                nn.Conv2d(1, 1, kernel_size=1, padding=0)
                # MaskedConv1D(embd_dim, embd_dim, 1,padding=0)
            )

        self.smooth = nn.ModuleList()
        for idx in range(arch[1] + 1):
            self.smooth.append(
                nn.Conv2d(1, 1, kernel_size=3, padding=1)
                # MaskedConv1D(embd_dim, embd_dim,3,padding=1)
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
            x: (Variable) top feature map to be upsampled.
            y: (Variable) lateral feature map.
        Returns:
            (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W)) + y

    def forward(self, x, mask):
        # print(x.shape)
        """
        Args:
            x (float tensor, (bs, c, t)): features.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        bs, _, t = x.size()
        # print(x.shape)
        # embedding projection
        if self.embd_fc is not None:
            x, _ = self.embd_fc(x, mask)
        # print(x.shape)
        # add position encoding
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
            pe = pe * mask.float()

        if pe is not None:
            x = x + pe

        # stem transformer
        for idx in range(len(self.stem)):
            x, _ = self.stem[idx](x, mask)

        # branch transformer
        fpn_feats, fpn_masks = (x,), (mask,)
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            # print(x.shape)
            fpn_feats += (x,)
            fpn_masks += (mask,)
        fpn_feats = list(fpn_feats)

        # # adding top-down connection
        # for index in range(len(fpn_feats)):
        #     back_index = len(fpn_feats) - 1 - index
        #     if index == 0:
        #         fpn_feats[back_index] = self.latlayers[back_index](fpn_feats[back_index].unsqueeze(1)).squeeze(1)
        #         continue
        #     else:
        #         temp_feature = self.latlayers[back_index](fpn_feats[back_index].unsqueeze(1))
        #         fpn_feats[back_index] = self.smooth[back_index](
        #             self._upsample_add(fpn_feats[back_index + 1].unsqueeze(1), temp_feature)).squeeze(1)
        #
        # fpn_feats = tuple(fpn_feats)

        return fpn_feats, fpn_masks


@register_backbone('video_transformer_1')
class VideoTransformer1(nn.Module):
    """
        A backbone that combines convolutions with transformer encoder layers
        to build a feature pyramid.
        [embedding convs] -> [stem transformers] -> [branch transformers]
    """

    def __init__(
            self,
            in_dim,  # video feature dimension
            embd_dim,  # embedding dimension
            max_seq_len,  # max sequence length
            arch=(2, 5),  # (#stem transformers, #branch transformers)
            n_heads=4,  # number of attention heads for MHA
            mha_win_size=-1,  # local window size for MHA
            attn_pdrop=0.1,  # dropout rate for attention maps
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            use_rel_pe=False,  # if True, use relative position encoding
            pe_type=0,  # position encoding type (-1, 0, 1)
            **kwargs,
    ):
        super(VideoTransformer1, self).__init__()

        assert len(arch) == 2, '(#stem transformers, #branch transformers)'

        self.max_seq_len = max_seq_len

        if isinstance(mha_win_size, int):
            mha_win_size = (mha_win_size,) * (arch[-1] + 1)
        assert isinstance(mha_win_size, (list, tuple))
        assert len(mha_win_size) == arch[-1] + 1
        # position encoding (c, t)
        assert pe_type in (-1, 0, 1)
        self.pe_type = pe_type
        if pe_type >= 0:
            pe = get_sinusoid_encoding(max_seq_len, embd_dim // 2, pe_type)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # embedding projection
        self.embd_fc = None
        if in_dim != embd_dim:
            self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)
        # stem transformer blocks
        self.stem = nn.ModuleList()
        for _ in range(arch[0]):
            self.stem.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=1,
                    n_heads=n_heads,
                    window_size=mha_win_size[0],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            )

        # branch transformer blocks
        self.branch = nn.ModuleList()
        for idx in range(arch[1]):
            self.branch.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=2,
                    n_heads=n_heads,
                    window_size=mha_win_size[idx + 1],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            )

        self.branch1 = nn.ModuleList()
        for idx in range(arch[1]):
            self.branch1.append(
                nn.AvgPool1d(2, stride=2)
            )

        # top-down branch transformer
        self.top_down = nn.ModuleList()
        self.top_down.append(
            TransformerEncoderBlock(
                embd_dim,
                stride=1,
                n_heads=n_heads,
                window_size=9,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                use_rel_pe=use_rel_pe,
            )

        )
        for idx in range(arch[1]):
            self.top_down.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=2,
                    n_heads=n_heads,
                    window_size=9,
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            )
        self.embd_fc_group = nn.ModuleList()
        for idx in range(arch[1] + 1):
            if idx <= 2:
                self.embd_fc_group.append(
                    MaskedConv1D(embd_dim * (idx + 1), embd_dim, 1)
                )
            else:
                self.embd_fc_group.append(
                    MaskedConv1D(embd_dim * (idx + 1), embd_dim, 1)
                )

        self.latlayers = nn.ModuleList()
        for idx in range(arch[1] + 1):
            self.latlayers.append(
                nn.Conv2d(1, 1, kernel_size=1, padding=0)
            )

        self.smooth = nn.ModuleList()
        for idx in range(arch[1] + 1):
            self.smooth.append(
                nn.Conv2d(1, 1, kernel_size=3, padding=1)
            )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        # print(x.shape)
        # for each in x:
        #     print(each.shape)
        """
        Args:
            x (float tensor, (bs, c, t)): features.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        bs, _, t = x.size()
        if self.embd_fc is not None:
            x, _ = self.embd_fc(x, mask)

        # add position encoding
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
            pe = pe * mask.float()

        if pe is not None:
            x = x + pe


        # branch transformer
        fpn_feats, fpn_masks = (x,), (mask,)
        for idx in range(len(self.branch)):
            x1 = x
            x1, mask = self.branch[idx](x1, mask)
            x = self.branch1[idx](x)
            fpn_feats += (x,)
            fpn_masks += (mask,)
        fpn_feats = list(fpn_feats)
        masks = fpn_masks

        limit = 3

        temp_features = []
        for i in range(len(fpn_feats)):
            temp_features.append([])

        for idx1 in range(len(fpn_feats)):
            for idx2 in range(idx1):
                temp_features[idx1].append(0)
        for idx1 in range(len(fpn_feats)):
            current_feature = fpn_feats[idx1]

            for idx2 in range(idx1, len(fpn_feats)):
                if idx2 < idx1 + limit:
                    current_mask = masks[idx2]
                    current_feature, _ = self.top_down[idx2 - idx1](current_feature, current_mask)
                    temp_features[idx1].append(current_feature)
                else:
                    current_mask = masks[idx2]
                    current_feature, _ = self.top_down[idx2 - idx1](current_feature, current_mask)
                    temp_features[idx1].append(current_feature)
        temp_features_1 = []

        for i in range(len(fpn_feats)):
            temp_features_1.append([])
            for j in range(len(temp_features[i])):
                if torch.is_tensor(temp_features[j][i]):
                    temp_features_1[i].append(temp_features[j][i])

        temp_features = temp_features_1
        temp_features_2 = []
        for i in range(len(temp_features)):
            if i != 0:
                current_feature = torch.cat(temp_features[i], dim=1)
            else:
                current_feature = temp_features[i][0]
            current_feature, _ = self.embd_fc_group[i](current_feature, fpn_masks[i])
            temp_features_2.append(current_feature)

        fpn_feats = tuple(temp_features_2)
        return fpn_feats, fpn_masks


@register_backbone('video_transformer_2')
class VideoTransformer2(nn.Module):
    """
        A backbone that combines convolutions with transformer encoder layers
        to build a plain feature pyramid.
        [embedding convs] -> [stem transformers] -> [branch transformers]
        -> [stem transformers] -> [branch transformers]
        https://arxiv.org/pdf/2203.16527.pdf
    """

    def __init__(
            self,
            in_dim,  # video feature dimension
            embd_dim,  # embedding dimension
            max_seq_len,  # max sequence length
            arch=(2, 5),  # (#stem transformers, #branch transformers)
            n_heads=4,  # number of attention heads for MHA
            mha_win_size=-1,  # local window size for MHA
            attn_pdrop=0.1,  # dropout rate for attention maps
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            use_rel_pe=False,  # if True, use relative position encoding
            pe_type=0,  # position encoding type (-1, 0, 1)
            **kwargs,
    ):
        super(VideoTransformer2, self).__init__()

        assert len(arch) == 2, '(#stem transformers, #branch transformers)'

        self.max_seq_len = max_seq_len

        if isinstance(mha_win_size, int):
            mha_win_size = (mha_win_size,) * (arch[-1] + 1)
        assert isinstance(mha_win_size, (list, tuple))
        assert len(mha_win_size) == arch[-1] + 1
        # position encoding (c, t)
        assert pe_type in (-1, 0, 1)
        self.pe_type = pe_type
        if pe_type >= 0:
            pe = get_sinusoid_encoding(max_seq_len, embd_dim // 2, pe_type)
            pe /= embd_dim ** 0.5
            self.register_buffer('pe', pe, persistent=False)
        else:
            self.pe = None

        # embedding projection
        self.embd_fc = None
        if in_dim != embd_dim:
            self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)
        # stem transformer blocks
        self.stem = nn.ModuleList()
        for _ in range(arch[0]):
            self.stem.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=1,
                    n_heads=n_heads,
                    window_size=mha_win_size[0],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            )

        # branch transformer blocks
        self.branch = nn.ModuleList()
        for idx in range(arch[1]):
            self.branch.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=2,
                    n_heads=n_heads,
                    window_size=mha_win_size[idx + 1],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            )


        self.stem_batch = nn.ModuleList()
        for _ in range(4):
            self.stem_batch.append(
                TransformerEncoderBlock(
                    embd_dim,
                    stride=1,
                    n_heads=n_heads,
                    window_size=mha_win_size[0],
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    use_rel_pe=use_rel_pe,
                )
            )

        # currently fixed to height 5
        self.fpn_batch = nn.ModuleList()

        self.fpn_batch.append(
            TransformerEncoderBlockTopDown(
                embd_dim,
                stride=4,
                n_heads=n_heads,
                window_size=mha_win_size[0],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                use_rel_pe=use_rel_pe,
            )
        )
        self.fpn_batch.append(
            TransformerEncoderBlockTopDown(
                embd_dim,
                stride=2,
                n_heads=n_heads,
                window_size=mha_win_size[0],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                use_rel_pe=use_rel_pe,
            )
        )
        self.fpn_batch.append(
            TransformerEncoderBlock(
                embd_dim,
                stride=1,
                n_heads=n_heads,
                window_size=mha_win_size[0],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                use_rel_pe=use_rel_pe,
            )
        )
        self.fpn_batch.append(
            TransformerEncoderBlock(
                embd_dim,
                stride=2,
                n_heads=n_heads,
                window_size=mha_win_size[0],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                use_rel_pe=use_rel_pe,
            )
        )
        self.fpn_batch.append(
            TransformerEncoderBlock(
                embd_dim,
                stride=4,
                n_heads=n_heads,
                window_size=mha_win_size[0],
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                use_rel_pe=use_rel_pe,
            )
        )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): features.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        bs, _, t = x.size()
        if self.embd_fc is not None:
            x, _ = self.embd_fc(x, mask)

        # add position encoding
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
            pe = pe * mask.float()

        if pe is not None:
            x = x + pe

        # stem transformer
        for idx in range(len(self.stem)):
            x, _ = self.stem[idx](x, mask)

        # branch transformers to get video_feature with (c, 1/4 * t)
        temp_mask = [mask]
        x1 = x
        mask1 = mask
        for idx in range(len(self.branch)):
            x1, mask1 = self.branch[idx](x1, mask1)
            if idx == 1:
                x = x1
                mask = mask1
            temp_mask.append(mask1)


        # more stem transformers
        for idx in range(len(self.stem_batch)):
            x, mask = self.stem_batch[idx](x, mask)


        # get the fpn feature
        fpn_feats, fpn_masks = tuple(), tuple()
        for idx in range(len(self.fpn_batch)):
            fpn_feat, fpn_mask = self.fpn_batch[idx](x, mask)
            fpn_feats += (fpn_feat,)
            fpn_masks += (fpn_mask,)

        return fpn_feats, fpn_masks


@register_backbone('video_convnet')
class VideoConvNet(nn.Module):
    """
        A fully convolutional backbone to build a feature pyramid.
        [embedding projection] --> [stem convs] --> [branch convs]
    """

    def __init__(
            self,
            in_dim,  # video feature dimension
            embd_dim,  # embedding dimension
            arch=(2, 5),  # (#stem convs, #branch convs)
            **kwargs,
    ):
        super(VideoConvNet, self).__init__()

        assert len(arch) == 2, '(#stem convs, #branch convs)'

        # embedding projection
        self.embd_fc = None
        if in_dim != embd_dim:
            self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)

        # stem convs
        self.stem = nn.ModuleList()
        for _ in range(arch[0]):
            self.stem.append(ConvBlock(embd_dim, 3, 1))

        # branch convs
        self.branch = nn.ModuleList()
        for _ in range(arch[1]):
            self.branch.append(ConvBlock(embd_dim, 3, 2))

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): features.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        # embedding projection
        if self.embd_fc is not None:
            x, _ = self.embd_fc(x, mask)

        # stem convs
        for idx in range(len(self.stem)):
            x, _ = self.stem[idx](x, mask)

        # branch convs
        fpn_feats, fpn_masks = (x,), (mask,)
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            fpn_feats += (x,)
            fpn_masks += (mask,)

        return fpn_feats, fpn_masks


def make_backbone(name, **kwargs):
    return backbones[name](**kwargs)