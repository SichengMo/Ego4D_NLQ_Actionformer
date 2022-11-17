import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .weight_init import trunc_normal_


class MaskedConvTrans1D(nn.Module):
    """
    Masked 1D Transpose convolution
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
    ):
        super(MaskedConvTrans1D, self).__init__()

        assert kernel_size % 2 == 1 and kernel_size // 2 == padding
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride, 1,1, groups, bias, dilation,padding_mode
        )
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        assert x.size(-1) % self.stride == 0
        # print(self.stride)
        # print(x.shape)
        x = self.conv(x)
        mask_float = mask.float()

        if self.stride > 1:
            mask_float = F.interpolate(
                mask_float,
                size=x.size(-1),
                mode='nearest',
            )

        x = x * mask_float
        mask = mask_float.bool()
        # print(x.shape)
        return x, mask


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
    ):
        super(MaskedConv1D, self).__init__()

        assert kernel_size % 2 == 1 and kernel_size // 2 == padding
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias, padding_mode
        )
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        assert x.size(-1) % self.stride == 0

        x = self.conv(x)
        mask_float = mask.float()

        if self.stride > 1:
            mask_float = F.interpolate(
                mask_float,
                size=x.size(-1),
                mode='nearest',
            )

        x = x * mask_float
        mask = mask_float.bool()

        return x, mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports input of size (bs, c, t)
    """

    def __init__(self, n_channels, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()

        self.n_channels = n_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
        """
        assert x.size(1) == self.n_channels

        # channel-wise normalization
        mu = torch.mean(x, dim=1, keepdim=True)
        x = x - mu
        sigma = torch.mean(x ** 2, dim=1, keepdim=True)
        x = x / torch.sqrt(sigma + self.eps)

        if self.affine:
            x = x * self.weight + self.bias

        return x


def get_sinusoid_encoding(seq_len, n_freqs, method=0):
    """
    Sinusoid position encoding
    """
    if method == 0:  # transformer (https://arxiv.org/abs/1706.03762)
        tics = torch.arange(seq_len)
        freqs = 10000 ** (torch.arange(n_freqs) / n_freqs)
        x = tics[None, :] / freqs[:, None]  # (n, t)
    else:  # perceiver (https://arxiv.org/abs/2103.03206)
        tics = (torch.arange(seq_len)) / seq_len * 2 - 1
        freqs = torch.linspace(1, seq_len / 2, n_freqs)
        x = math.pi * freqs[:, None] * tics[None, :]  # (n, t)
    pe = torch.cat([torch.sin(x), torch.cos(x)])  # (n * 2, t)

    return pe


class MaskedMHA(nn.Module):
    """
    Multi Head Attention with mask
    NOTE: This implementation supports both self-attention and cross-attention.

    Modified from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            q_dim=None,  # query dimension
            kv_dim=None,  # key / value dimension
            out_dim=None,  # output dimension
            n_heads=4,  # number of attention heads
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
    ):
        super(MaskedMHA, self).__init__()

        assert embd_dim % n_heads == 0
        self.embd_dim = embd_dim

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim
        if out_dim is None:
            out_dim = q_dim

        self.n_heads = n_heads
        self.n_channels = embd_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.n_channels)

        self.query = nn.Conv1d(q_dim, embd_dim, 1)
        self.key = nn.Conv1d(kv_dim, embd_dim, 1)
        self.value = nn.Conv1d(kv_dim, embd_dim, 1)
        self.proj = nn.Conv1d(embd_dim, out_dim, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

    def forward(self, q, q_mask, k=None, v=None, kv_mask=None):
        """
        Args:
            q (float tensor, (bs, c, t1)): query feature sequence.
            q_mask (bool tensor, (bs, 1, t1)): query mask (1 for valid positions).
            k (float tensor, (bs, c, t2)): key feature sequence.
            v (float tensor, (bs, c, t2)): value feature sequence.
            kv_mask (bool tensor, (bs, 1, t2)): key / value mask (1 for valid positions).
        """
        bs, c = q.size(0), self.embd_dim
        h, d = self.n_heads, self.n_channels

        if k is None:
            k = q
        if v is None:
            v = q
        if kv_mask is None:
            kv_mask = q_mask

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(bs, h, d, -1).transpose(2, 3)  # (bs, h, t1, d)
        k = k.view(bs, h, d, -1)  # (bs, h, d, t2)
        v = v.view(bs, h, d, -1).transpose(2, 3)  # (bs, h, t2, d)

        attn = (q * self.scale) @ k  # (bs, h, t1, t2)
        attn = attn.masked_fill(
            torch.logical_not(kv_mask[:, :, None, :]), float('-inf')
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        q = attn @ v  # (bs, h, t1, d)

        q = q.transpose(2, 3).contiguous().view(bs, c, -1)  # (bs, c, t1)
        q = self.proj_drop(self.proj(q)) * q_mask.float()

        return q, q_mask


class LocalMaskedMHA(nn.Module):
    """
    Local Multi Head Attention with mask
    NOTE: This implementation only supports self-attention.

    The implementation is fairly tricky, modified from
    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            q_dim=None,  # query dimension
            kv_dim=None,  # key / value dimension
            out_dim=None,  # output dimension
            n_heads=4,  # number of attention heads
            window_size=9,  # size of the local attention window
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            use_rel_pe=False,  # if True, use relative position encoding
    ):
        super(LocalMaskedMHA, self).__init__()

        assert embd_dim % n_heads == 0
        self.embd_dim = embd_dim

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim
        if out_dim is None:
            out_dim = q_dim

        self.n_heads = n_heads
        self.n_channels = embd_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.n_channels)

        assert window_size % 2 == 1
        self.window_size = window_size
        self.stride = window_size // 2

        self.query = nn.Conv1d(q_dim, embd_dim, 1)
        self.key = nn.Conv1d(kv_dim, embd_dim, 1)
        self.value = nn.Conv1d(kv_dim, embd_dim, 1)
        self.proj = nn.Conv1d(embd_dim, out_dim, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        # masks for local attention (left / right paddings)
        l_mask = torch.ones(self.stride, self.stride + 1).tril().flip(dims=(0,))
        r_mask = torch.ones(self.stride, self.stride + 1).tril().flip(dims=(1,))
        self.register_buffer('l_mask', l_mask.bool(), persistent=False)
        self.register_buffer('r_mask', r_mask.bool(), persistent=False)

        self.use_rel_pe = use_rel_pe
        if use_rel_pe:
            self.rel_pe = nn.Parameter(torch.zeros(n_heads, 1, window_size))
            trunc_normal_(self.rel_pe, std=(2.0 / embd_dim) ** 0.5)

    def _chunk(self, x, size):
        """
        Convert feature sequence into temporally overlapping chunks.

        Args:
            x (float tensor, (n, t, d)): feature sequence.
            size (int): chunk size.

        Returns:
            x (float tensor, (n, k, s, d)): chunked features.
        """
        n, t, d = x.size()
        assert (t + self.stride - size) % self.stride == 0
        n_chunks = (t + self.stride - size) // self.stride

        chunk_size = (n, n_chunks, size, d)
        chunk_stride = (x.stride(0), self.stride * x.stride(1), *x.stride()[1:])
        x = x.as_strided(size=chunk_size, stride=chunk_stride)

        return x

    def _query_key_matmul(self, q, k):
        """
        Chunk-wise query-key product.

        Args:
            q (float tensor, (n, t, d)): query tensor.
            k (float tensor, (n, t, d)): key tensor.

        Returns:
            attn (float tensor, (n, t, w)): unnormalized attention scores.
        """
        assert q.size() == k.size()
        n, t, _ = q.size()
        w, s = self.window_size, self.stride

        # chunk query and key tensors: (n, t, d) -> (n, t // s - 1, 2s, d)
        q_chunks = self._chunk(q.contiguous(), size=2 * s)
        k_chunks = self._chunk(k.contiguous(), size=2 * s)
        n_chunks = q_chunks.size(1)

        # chunk-wise attention scores: (n, t // s - 1, 2s, 2s)
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (q_chunks, k_chunks))

        # shift diagonals into columns: (n, t // s - 1, 2s, w)
        chunk_attn = F.pad(chunk_attn, (0, 0, 0, 1))
        chunk_attn = chunk_attn.view(n, n_chunks, 2 * s, w)

        # fill in the overall attention matrix: (n, t // s, s, w)
        attn = chunk_attn.new_zeros(n, t // s, s, w)
        attn[:, :-1, :, s:] = chunk_attn[:, :, :s, :s + 1]
        attn[:, -1, :, s:] = chunk_attn[:, -1, s:, :s + 1]
        attn[:, 1:, :, :s] = chunk_attn[:, :, -(s + 1):-1, s + 1:]
        attn[:, 0, 1:s, 1:s] = chunk_attn[:, 0, :s - 1, -(s - 1):]
        attn = attn.view(n, t, w)

        # mask invalid attention scores
        attn[:, :s, :s + 1].masked_fill_(self.l_mask, float('-inf'))
        attn[:, -s:, -(s + 1):].masked_fill_(self.r_mask, float('-inf'))

        return attn

    def _attn_normalize(self, attn, mask):
        """
        Normalize attention scores over valid positions.

        Args:
            attn (float tensor, (bs, h, t, w)): unnormalized attention scores.
            mask (bool tensor, (bs, t, 1)): mask (1 for valid positions).

        Returns:
            attn (float tensor, (bs, h, t, w)): normalized attention map.
        """
        bs, h, t, w = attn.size()

        # inverse mask (0 for valid positions, -inf for invalid ones)
        inv_mask = torch.logical_not(mask)
        inv_mask_float = inv_mask.float().masked_fill(inv_mask, -1e4)

        # additive attention mask: (bs, t, w)
        attn_mask = self._query_key_matmul(
            torch.ones_like(inv_mask_float), inv_mask_float
        )
        attn += attn_mask.view(bs, 1, t, w)

        # normalize
        attn = F.softmax(attn, dim=-1)

        # if all key / value positions in a local window are invalid
        # (i.e., when the query position is invalid), softmax returns NaN.
        # Replace NaNs with 0
        attn = attn.masked_fill(inv_mask.unsqueeze(1), 0.0)

        return attn

    def _attn_value_matmul(self, attn, v):
        """
        Chunk-wise attention-value product.

        Args:
            attn (float tensor, (n, t, w)): attention map.
            v (float tensor, (n, t, d)): value tensor.

        Returns:
            out (float tensor, (n, t, d)): attention-weighted sum of values.
        """
        n, t, d = v.size()
        w, s = self.window_size, self.stride

        # chunk attention map: (n, t, w) -> (n, t // s, s, w)
        attn_chunks = attn.view(n, t // s, s, w)

        # shift columns into diagonals: (n, t // s, s, 3s)
        attn_chunks = F.pad(attn_chunks, (0, s))
        attn_chunks = attn_chunks.view(n, t // s, -1)[..., :-s]
        attn_chunks = attn_chunks.view(n, t // s, s, 3 * s)

        # chunk value tensor: (n, t + 2s, d) -> (n, t // s, 3s, d)
        v = F.pad(v, (0, 0, s, s))
        v_chunks = self._chunk(v.contiguous(), size=3 * s)

        # chunk-wise attention-weighted sum: (n, t // s, s, d)
        out = torch.einsum('bcwd,bcdh->bcwh', (attn_chunks, v_chunks))
        out = out.view(n, t, d)

        return out

    def forward(self, q, q_mask, k=None, v=None, kv_mask=None):
        """
        Args:
            q (float tensor, (bs, c, t1)): query feature sequence.
            q_mask (bool tensor, (bs, 1, t1)): query mask (1 for valid positions).
            k (float tensor, (bs, c, t2)): key feature sequence.
            v (float tensor, (bs, c, t2)): value feature sequence.
            kv_mask (bool tensor, (bs, 1, t2)): key / value mask (1 for valid positions).
        """
        bs, c = q.size(0), self.embd_dim
        h, d, w = self.n_heads, self.n_channels, self.window_size

        if k is None:
            k = q
        if v is None:
            v = q
        if kv_mask is None:
            kv_mask = q_mask

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        q = q.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)  # (bs * h, t, d)
        k = k.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)  # (bs * h, t, d)
        v = v.view(bs, h, d, -1).flatten(0, 1).transpose(1, 2)  # (bs * h, t, d)

        # attention scores: (bs * h, t, w)
        attn = self._query_key_matmul(q * self.scale, k)
        attn = attn.view(bs, h, -1, w)
        if self.use_rel_pe:
            attn += self.rel_pe

        # normalized attention map: (bs, h, t, w)
        attn = self._attn_normalize(attn, kv_mask.transpose(1, 2))
        attn = self.attn_drop(attn)
        attn = attn.view(bs * h, -1, w)

        # attention-weighted sum of values: # (bs * h, t, d)
        q = self._attn_value_matmul(attn, v)
        q = q.view(bs, h, -1, d)

        q = q.transpose(2, 3).contiguous().view(bs, c, -1)
        q = self.proj_drop(self.proj(q)) * q_mask.float()

        return q, q_mask


class MaskedMHCA(nn.Module):
    """
    Multi Head Conv Attention with mask
    NOTE: This implementation only supports self-attention.

    Add a depth-wise convolution within a standard MHA
    (1) encode relative position information (replacing position encoding);
    (2) down-sample features if needed.

    Note: With current implementation, the downpsampled features will be aligned
    with every s+1 time steps, where s is the down-sampling stride. This allows us
    to easily interpolate the corresponding position encoding.
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            q_dim=None,  # query dimension
            kv_dim=None,  # key / value dimension
            out_dim=None,  # output dimension
            stride=1,  # convolution stride
            n_heads=4,  # number of attention heads
            window_size=-1,  # window size for local attention
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            use_rel_pe=False,  # if True, use relative position encoding
    ):
        super(MaskedMHCA, self).__init__()

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim

        self.use_conv = stride > 0
        if self.use_conv:
            # depth-wise convs
            assert stride == 1 or stride % 2 == 0
            kernel_size = stride + 1 if stride > 1 else 3

            self.query_conv = MaskedConv1D(
                q_dim, q_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=q_dim,
                bias=False,
            )
            self.key_conv = MaskedConv1D(
                kv_dim, kv_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=kv_dim,
                bias=False,
            )
            self.value_conv = MaskedConv1D(
                kv_dim, kv_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=kv_dim,
                bias=False,
            )

            # post-conv layernorms
            self.query_norm = LayerNorm(q_dim)
            self.key_norm = LayerNorm(kv_dim)
            self.value_norm = LayerNorm(kv_dim)

        # attention
        if window_size > 0:
            self.attn = LocalMaskedMHA(
                embd_dim,
                q_dim,
                kv_dim,
                out_dim,
                n_heads=n_heads,
                window_size=window_size,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe,
            )
        else:
            self.attn = MaskedMHA(
                embd_dim,
                q_dim,
                kv_dim,
                out_dim,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        if self.use_conv:
            q, q_mask = self.query_conv(x, mask)
            k, kv_mask = self.key_conv(x, mask)
            v, _ = self.value_conv(x, mask)

            q = self.query_norm(q)
            k = self.key_norm(k)
            v = self.value_norm(v)
        else:
            q = k = v = x
            q_mask = kv_mask = mask

        x, mask = self.attn(q, q_mask, k, v, kv_mask)

        return x, mask


class MaskedMHCTA(nn.Module):
    """
    Multi Head Transpose Conv Attention with mask
    NOTE: This implementation only supports self-attention.


    Masked Multi Head Conv Transpose Attention


    Add a depth-wise Transpose convolution within a standard MHA
    (1) encode relative position information (replacing position encoding);
    (2) down-sample features if needed.

    Note: With current implementation, the downpsampled features will be aligned
    with every s+1 time steps, where s is the down-sampling stride. This allows us
    to easily interpolate the corresponding position encoding.
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            q_dim=None,  # query dimension
            kv_dim=None,  # key / value dimension
            out_dim=None,  # output dimension
            stride=1,  # convolution stride
            n_heads=4,  # number of attention heads
            window_size=-1,  # window size for local attention
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            use_rel_pe=False,  # if True, use relative position encoding
    ):
        super(MaskedMHCTA, self).__init__()

        if q_dim is None:
            q_dim = embd_dim
        if kv_dim is None:
            kv_dim = embd_dim

        self.use_conv = stride > 0
        if self.use_conv:
            # depth-wise convs
            #assert stride == 1 or stride % 2 == 0
            kernel_size = stride + 1 if stride > 1 else 3

            self.query_conv = MaskedConvTrans1D(
                q_dim, q_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=q_dim,
                bias=False,
            )
            self.key_conv = MaskedConvTrans1D(
                kv_dim, kv_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=kv_dim,
                bias=False,
            )
            self.value_conv = MaskedConvTrans1D(
                kv_dim, kv_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=kv_dim,
                bias=False,
            )

            # post-conv layernorms
            self.query_norm = LayerNorm(q_dim)
            self.key_norm = LayerNorm(kv_dim)
            self.value_norm = LayerNorm(kv_dim)

        # attention
        if window_size > 0:
            self.attn = LocalMaskedMHA(
                embd_dim,
                q_dim,
                kv_dim,
                out_dim,
                n_heads=n_heads,
                window_size=window_size,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe,
            )
        else:
            self.attn = MaskedMHA(
                embd_dim,
                q_dim,
                kv_dim,
                out_dim,
                n_heads=n_heads,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
            )

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        if self.use_conv:
            q, q_mask = self.query_conv(x, mask)
            k, kv_mask = self.key_conv(x, mask)
            v, _ = self.value_conv(x, mask)

            q = self.query_norm(q)
            k = self.key_norm(k)
            v = self.value_norm(v)
        else:
            q = k = v = x
            q_mask = kv_mask = mask

        x, mask = self.attn(q, q_mask, k, v, kv_mask)

        return x, mask


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer.
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            stride=-1,  # convolution stride (-1 if disable convs)
            n_heads=4,  # number of attention heads
            window_size=-1,  # MHA window size (-1 for global attention)
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            out_ln=True,  # if True, apply layernorm on output
            use_rel_pe=False,  # if True, use relative position encoding
    ):
        super(TransformerEncoderLayer, self).__init__()

        # self-attention
        self.attn = MaskedMHCA(
            embd_dim,
            stride=stride,
            n_heads=n_heads,
            window_size=window_size,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            use_rel_pe=use_rel_pe,
        )
        self.ln1 = LayerNorm(embd_dim)

        # residual connection
        self.attn_skip = nn.Identity()
        if stride > 1:
            self.attn_skip = nn.MaxPool1d(
                kernel_size=stride + 1,
                stride=stride,
                padding=(stride + 1) // 2,
            )

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv1d(embd_dim, embd_dim * 4, 1),
            nn.GELU(),
            nn.Dropout(proj_pdrop),
            nn.Conv1d(embd_dim * 4, embd_dim, 1),
            nn.Dropout(proj_pdrop),
        )
        self.ln2 = LayerNorm(embd_dim) if out_ln else nn.Identity()

        self.drop_path_attn = self.drop_path_ffn = nn.Identity()
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(embd_dim, drop_prob=path_pdrop)
            self.drop_path_ffn = AffineDropPath(embd_dim, drop_prob=path_pdrop)

    def forward(self, x, mask, pe=None):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        if pe is not None:
            x = x + pe

        # self-attention (optionally with conv)
        dx, mask = self.attn(x, mask)
        mask_float = mask.float()
        x = self.attn_skip(x) * mask_float + self.drop_path_attn(dx)
        x = self.ln1(x)

        # FFN
        dx = self.ffn(x) * mask_float
        x = x + self.drop_path_ffn(dx)
        x = self.ln2(x)

        return x, mask

class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block.
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            stride=-1,  # convolution stride (-1 if disable convs)
            n_heads=4,  # number of attention heads
            window_size=-1,  # MHA window size (-1 for global attention)
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            out_ln=True,  # if True, apply layernorm on output
            use_rel_pe=False,  # if True, use relative position encoding
    ):
        super(TransformerEncoderBlock, self).__init__()

        # self-attention
        self.attn = MaskedMHCA(
            embd_dim,
            stride=stride,
            n_heads=n_heads,
            window_size=window_size,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            use_rel_pe=use_rel_pe,
        )
        self.ln1 = LayerNorm(embd_dim)

        # residual connection
        self.attn_skip = nn.Identity()
        if stride > 1:
            self.attn_skip = nn.MaxPool1d(
                kernel_size=stride + 1,
                stride=stride,
                padding=(stride + 1) // 2,
            )

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv1d(embd_dim, embd_dim * 4, 1),
            nn.GELU(),
            nn.Dropout(proj_pdrop),
            nn.Conv1d(embd_dim * 4, embd_dim, 1),
            nn.Dropout(proj_pdrop),
        )
        self.ln2 = LayerNorm(embd_dim) if out_ln else nn.Identity()

        self.drop_path_attn = self.drop_path_ffn = nn.Identity()
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(embd_dim, drop_prob=path_pdrop)
            self.drop_path_ffn = AffineDropPath(embd_dim, drop_prob=path_pdrop)

    def forward(self, x, mask, pe=None):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        if pe is not None:
            x = x + pe
        # self-attention (optionally with conv)
        dx, mask = self.attn(x, mask)
        mask_float = mask.float()
        x = self.attn_skip(x) * mask_float + self.drop_path_attn(dx)
        x = self.ln1(x)

        # FFN
        dx = self.ffn(x) * mask_float
        x = x + self.drop_path_ffn(dx)
        x = self.ln2(x)

        return x, mask


class TransformerEncoderBlockTopDown(nn.Module):
    """
    Transformer encoder block with transpose Convlution block.
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            stride=-1,  # convolution stride (-1 if disable convs)
            n_heads=4,  # number of attention heads
            window_size=-1,  # MHA window size (-1 for global attention)
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            out_ln=True,  # if True, apply layernorm on output
            use_rel_pe=False,  # if True, use relative position encoding
    ):
        super(TransformerEncoderBlockTopDown, self).__init__()

        # self-attention
        self.attn = MaskedMHCTA(
            embd_dim,
            stride=stride,
            n_heads=n_heads,
            window_size=window_size,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            use_rel_pe=use_rel_pe,
        )
        self.ln1 = LayerNorm(embd_dim)

        # residual connection
        self.attn_skip = nn.Identity()
        if stride < 1:
            self.attn_skip = nn.MaxPool1d(
                kernel_size=stride + 1,
                stride=stride,
                padding=(stride + 1) // 2,
            )
        elif stride >1:
            self.attn_skip = nn.Upsample(
                scale_factor=stride,
                mode = 'nearest',
            )

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv1d(embd_dim, embd_dim * 4, 1),
            nn.GELU(),
            nn.Dropout(proj_pdrop),
            nn.Conv1d(embd_dim * 4, embd_dim, 1),
            nn.Dropout(proj_pdrop),
        )
        self.ln2 = LayerNorm(embd_dim) if out_ln else nn.Identity()

        self.drop_path_attn = self.drop_path_ffn = nn.Identity()
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(embd_dim, drop_prob=path_pdrop)
            self.drop_path_ffn = AffineDropPath(embd_dim, drop_prob=path_pdrop)

    def forward(self, x, mask, pe=None):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        if pe is not None:
            x = x + pe

        # self-attention (optionally with conv)
        dx, mask = self.attn(x, mask)
        mask_float = mask.float()
        x = self.attn_skip(x) * mask_float + self.drop_path_attn(dx)
        x = self.ln1(x)

        # FFN
        dx = self.ffn(x) * mask_float
        x = x + self.drop_path_ffn(dx)
        x = self.ln2(x)

        return x, mask


class TransformerDecoderBlock(nn.Module):
    """
    Transformer decoder block.
    """

    def __init__(
            self,
            embd_dim,  # embedding dimension
            stride=-1,  # convolution stride (-1 if disable convs)
            n_heads=4,  # number of attention heads
            window_size=-1,  # MHA window size
            attn_pdrop=0.1,  # dropout rate for attention map
            proj_pdrop=0.1,  # dropout rate for projection
            path_pdrop=0.1,  # dropout rate for residual paths
            out_ln=True,  # if True, apply layernorm on output
            use_rel_pe=False,  # if True, use relative position encoding
            use_self_attn=True,  # if True, use self-attention
    ):
        super(TransformerDecoderBlock, self).__init__()

        self.use_self_attn = use_self_attn
        if use_self_attn:
            # self-attention
            self.attn = MaskedMHCA(
                embd_dim,
                stride=stride,
                n_heads=n_heads,
                window_size=window_size,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                use_rel_pe=use_rel_pe,
            )
            self.ln1 = LayerNorm(embd_dim)

            # residual connection
            self.attn_skip = nn.Identity()
            if stride > 1:
                self.attn_skip = nn.MaxPool1d(
                    kernel_size=stride + 1,
                    stride=stride,
                    padding=(stride + 1) // 2,
                )

        # cross-attention
        self.xattn = MaskedMHA(
            embd_dim,
            n_heads=n_heads,
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
        )
        self.ln2 = LayerNorm(embd_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Conv1d(embd_dim, embd_dim * 4, 1),
            nn.GELU(),
            nn.Dropout(proj_pdrop),
            nn.Conv1d(embd_dim * 4, embd_dim, 1),
            nn.Dropout(proj_pdrop),
        )
        self.ln3 = LayerNorm(embd_dim) if out_ln else nn.Identity()

        self.drop_path_attn = self.drop_path_ffn = nn.Identity()
        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(embd_dim, drop_prob=path_pdrop)
            self.drop_path_ffn = AffineDropPath(embd_dim, drop_prob=path_pdrop)

    def forward(self, q, q_mask, kv, kv_mask, q_pe=None, kv_pe=None):
        """
        Args:
            q (float tensor, (bs, c, t1)): query feature sequence.
            q_mask (bool tensor, (bs, 1, t1)): query mask (1 for valid positions).
            kv (float tensor, (bs, c, t2)): key / value feature sequence.
            kv_mask (bool tensor, (bs, 1, t2)): key / value mask (1 for valid positions).
        """
        if q_pe is not None:
            q = q + q_pe
        if kv_pe is not None:
            kv = kv + kv_pe

        # self-attention
        if self.use_self_attn:
            dq, _ = self.attn(q, q_mask)
            q = self.attn_skip(q) + self.drop_path_attn(dq)
            q = self.ln1(q)

        # cross-attention
        dq, _ = self.xattn(q, q_mask, kv, kv, kv_mask)
        q = q + self.drop_path_attn(dq)
        q = self.ln2(q)

        # FFN
        dq = self.ffn(q) * q_mask.float()
        q = q + self.drop_path_ffn(dq)
        q = self.ln3(q)

        return q, q_mask


class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            in_dim,  # input dimension
            embd_dim=None,  # embedding dimension
            out_dim=None,  # output dimension
            stride=1,  # downsampling stride
            expansion=2,  # expansion factor for residual block
            padding=0,
    ):
        super(ConvBlock, self).__init__()

        if embd_dim is None:
            embd_dim = in_dim
        if out_dim is None:
            out_dim = in_dim

        width = embd_dim * expansion
        self.conv1 = MaskedConv1D(
            in_dim, width, 3, stride, padding=padding
        )
        self.conv2 = MaskedConv1D(
            width, out_dim, 3, 1, padding=padding
        )

        self.downsample = None
        if stride > 1 or out_dim != in_dim:
            self.downsample = MaskedConv1D(in_dim, out_dim, 1, stride)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        dx, dx_mask = self.conv1(x, mask)
        dx = F.relu(dx, inplace=True)
        dx, _ = self.conv2(dx, dx_mask)

        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        x = F.relu(x + dx, inplace=True)

        return x, mask


class MaskedInstanceNorm1D(nn.Module):
    """
    Masked 1D instance normalization.
    """

    def __init__(self, n_channels, eps=1e-5, affine=False):
        super(MaskedInstanceNorm1D, self).__init__()

        self.n_channels = n_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(1, n_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, n_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x, mask):
        """
        Args:
            x (float tensor, (bs, c, t)): feature sequence.
            mask (bool tensor, (bs, 1, t)): mask (1 for valid positions).
        """
        mask_float = mask.float()

        x = x * mask_float
        n_valid = torch.sum(mask_float, dim=-1, keepdim=True)
        mean = torch.sum(x, dim=-1, keepdim=True) / n_valid
        var = torch.sum(x ** 2, dim=-1, keepdim=True) / n_valid - mean ** 2
        x = (x - mean) / torch.sqrt(F.relu(var) + self.eps)

        if self.affine:
            x = self.weight * x + self.bias

        x = x * mask_float

        return x, mask

class MaskedAdaAttN(nn.Module):
    """
    Adaptive attention normalization: https://arxiv.org/abs/2108.03647
    """

    def __init__(
            self,
            q_dim,  # query dimension
            kv_dim=None,  # key / value dimension
            n_heads=4,  # number of attention heads
            attn_pdrop=0.1,  # dropout rate for attention map
            norm_type='in',  # normalization method ('in', 'ln', 'none')
    ):
        super(MaskedAdaAttN, self).__init__()

        assert q_dim % n_heads == 0
        if kv_dim is None:
            kv_dim = q_dim

        self.n_heads = n_heads
        self.n_channels = q_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.n_channels)

        self.query = nn.Conv1d(q_dim, q_dim, 1)
        self.key = nn.Conv1d(kv_dim, q_dim, 1)
        self.value = nn.Conv1d(kv_dim, q_dim, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)

        assert norm_type in ('in', 'ln', 'none')
        self.norm_type = norm_type
        if norm_type == 'in':
            self.norm = MaskedInstanceNorm1D(q_dim, affine=False)
        elif norm_type == 'ln':
            self.norm = LayerNorm(q_dim, affine=False)
        else:
            self.norm = nn.Identity()

    def forward(self, x, mask, kv, kv_mask):
        """
        Args:
            x (float tensor, (bs, c, t1)): query feature sequence.
            mask (bool tensor, (bs, 1, t1)): query mask (1 for valid positions).
            kv (float tensor, (bs, c, t2)): key / value feature.
            kv_mask (bool tensor, (bs, 1, t2)): key / value mask (1 for valid positions).
        """
        bs, c, _ = x.size()
        h, d = self.n_heads, self.n_channels

        q = self.query(x)
        k = self.key(kv)
        v = self.value(kv)

        q = q.view(bs, h, d, -1).transpose(2, 3)
        k = k.view(bs, h, d, -1)
        v = v.view(bs, h, d, -1).transpose(2, 3)

        # self-attention
        attn = (q * self.scale) @ k
        attn = attn.masked_fill(
            torch.logical_not(kv_mask[:, :, None, :]), float('-inf')
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # attention-weighted statistics
        mean = attn @ v  # (bs, h, t1, d)
        var = F.relu(attn @ v ** 2 - mean ** 2)  # (bs, h, t1, d)
        std = torch.sqrt(var)

        mean = mean.transpose(2, 3).contiguous().view(bs, c, -1)
        std = std.transpose(2, 3).contiguous().view(bs, c, -1)

        # adaptive normalization
        if self.norm_type == 'in':
            x, _ = self.norm(x, mask)
        else:
            x = self.norm(x)
        x = x * std + mean

        x = x * mask.float()

        return x, mask


class MaskedCrossAtten(nn.Module):
    def __init__(
            self,
            q_dim,  # query dimension
            kv_dim=None,  # key / value dimension
            n_heads=4,  # number of attention heads
            attn_pdrop=0.1,  # dropout rate for attention map
            norm_type='in',  # normalization method ('in', 'ln', 'none')
    ):
        super(MaskedCrossAtten, self).__init__()

        assert q_dim % n_heads == 0
        if kv_dim is None:
            kv_dim = q_dim

        self.n_heads = n_heads
        self.n_channels = q_dim // n_heads
        self.scale = 1.0 / math.sqrt(self.n_channels)

        self.query = nn.Conv1d(q_dim, q_dim, 1)
        self.key = nn.Conv1d(kv_dim, q_dim, 1)
        self.value = nn.Conv1d(kv_dim, q_dim, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)

        assert norm_type in ('in', 'ln', 'none')
        self.norm_type = norm_type
        if norm_type == 'in':
            self.norm = MaskedInstanceNorm1D(q_dim, affine=False)
        elif norm_type == 'ln':
            self.norm = LayerNorm(q_dim, affine=False)
        else:
            self.norm = nn.Identity()
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x, mask, kv, kv_mask):
        """
        Args:
            x (float tensor, (bs, c, t1)): query feature sequence.
            mask (bool tensor, (bs, 1, t1)): query mask (1 for valid positions).
            kv (float tensor, (bs, c, t2)): key / value feature.
            kv_mask (bool tensor, (bs, 1, t2)): key / value mask (1 for valid positions).
        """

        temp_x = x
        bs, c, _ = x.size()
        h, d = self.n_heads, self.n_channels
        q = self.query(x)
        k = self.key(kv)
        v = self.value(kv)
        q = q.view(bs, h, d, -1).transpose(2, 3)
        k = k.view(bs, h, d, -1)
        v = v.view(bs, h, d, -1).transpose(2, 3)

        # cross-attention
        attn = (q * self.scale) @ k
        attn = attn.masked_fill(
            torch.logical_not(kv_mask[:, :, None, :]), float('-inf')
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(2, 3).contiguous().view(bs, c, -1)
        x = x + temp_x

        return x, mask

class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, exp=False):
        super(Scale, self).__init__()

        self.exp = exp
        init = 0.0 if exp else 1.0
        self.scale = nn.Parameter(torch.as_tensor(init, dtype=torch.float))

    def forward(self, x):
        if self.exp:
            x = x * torch.exp(self.scale)
        else:
            x = x * self.scale
        return x


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    x = x.div(keep_prob) * mask

    return x


class DropPath(nn.Module):
    """
    Drop paths per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths per sample (when applied in main path of residual blocks)
    with a per channel scaling factor (and zero init).

    https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, dim, drop_prob=0.0, init_scale=1e-4):
        super(AffineDropPath, self).__init__()

        self.scale = nn.Parameter(init_scale * torch.ones((1, dim, 1)))
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)