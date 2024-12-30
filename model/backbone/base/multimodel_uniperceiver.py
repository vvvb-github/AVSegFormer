from .uniperceiver import VisualPatchEmbedding, TokenBaseEmbedding
import logging
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.runner import load_checkpoint
from mmdet.utils import get_root_logger
from timm.models.layers import DropPath
from torch import nn


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()\
        .view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def batch_mean(x, batch_size):
    """
    Args:
        x (torch.Tensor): shape=(B,N,C)
        batch_size (int): target batch size

    Returns:
        s: shape=(batch_size,N,C)
    """
    B, N, C = x.shape
    assert B % batch_size == 0
    Nw = B // batch_size
    x = x.reshape(Nw, batch_size, N, C)
    x = torch.mean(x, dim=0)
    return x.reshape(batch_size, N, C)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.in_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, q, q_mask, H, W):
        # mask.shape = [Bq, Nq]
        B, N, C = x.shape
        Bq, Nq, Cq = q.shape
        assert B == Bq
        x = torch.cat([x, q], dim=1)

        qkv = self.in_proj(x).reshape(
            B, N+Nq, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)  # [B, num_heads, N+Nq, C // num_heads]

        attn = (q @ k.transpose(-2, -1)) * \
            self.scale  # [B, num_heads, N+Nq, N+Nq]

        if self.use_mask:
            mask = torch.cat(
                [torch.ones([B, N], dtype=q_mask.dtype).cuda(), q_mask], dim=1)
            mask = mask.reshape(B, 1, 1, N+Nq)
            attn = attn.masked_fill(mask == 0, -1e18)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N+Nq, C)

        x = self.out_proj(x)
        x = self.proj_drop(x)
        q = x[:, N:, :]
        x = x[:, :N, :]

        return x, q


class WindowedAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 window_size=14, use_mask=True):
        super().__init__()
        self.use_mask = use_mask
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.in_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.window_size = window_size

    def forward(self, x, q, q_mask, H, W):
        B, N, C = x.shape
        Bq, Nq, Cq = q.shape
        assert B == Bq

        N_ = self.window_size * self.window_size
        H_ = math.ceil(H / self.window_size) * self.window_size
        W_ = math.ceil(W / self.window_size) * self.window_size
        x = x.view(B, H, W, C)
        x = F.pad(x, [0, 0, 0, W_ - W, 0, H_ - H])

        # nW*B, window_size, window_size, C
        x = window_partition(x, window_size=self.window_size)
        x = x.view(-1, N_, C)

        B_ = x.shape[0]
        q = q.unsqueeze(0).expand(B_//Bq, Bq, Nq, Cq).reshape(B_, Nq, Cq)
        x = torch.cat([x, q], dim=1)

        qkv = self.in_proj(x).view(-1, N_+Nq, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * \
            self.scale  # [B, L, num_head, N_, N_]

        # mask
        if self.use_mask:
            mask = torch.cat(
                [torch.ones([B, N_], dtype=q_mask.dtype).cuda(), q_mask], dim=1)
            mask = mask.reshape(B, 1, 1, N_+Nq)
            mask = mask.unsqueeze(0).expand(
                B_//B, B, 1, 1, N_+Nq).reshape(B_, 1, 1, N_+Nq)
            attn = attn.masked_fill(mask == 0, -1e18)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, L, num_head, N_, N_]
        x = (attn @ v).transpose(1, 2).reshape(-1, N_+Nq, C)

        i = x[:, :N_, :].reshape(-1, self.window_size, self.window_size, C)
        t = batch_mean(x[:, N_:, :], Bq)

        x = window_reverse(i, self.window_size, H_, W_)
        x = x[:, :H, :W, :].reshape(B, N, C).contiguous()
        x = self.out_proj(x)
        x = self.proj_drop(x)
        q = self.out_proj(t)
        q = self.proj_drop(q)

        return x, q


class MultiModelBertLayer(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, num_attention_heads=12,
                 drop_path_ratio=0.1, windowed=False, window_size=14, with_cp=False, use_mask=True):

        super(MultiModelBertLayer, self).__init__()
        self.with_cp = with_cp
        if windowed:
            self.self_attn = WindowedAttention(hidden_size, num_attention_heads, qkv_bias=True, attn_drop=0.,
                                               proj_drop=0., window_size=window_size, use_mask=use_mask)
        else:
            self.self_attn = Attention(hidden_size, num_attention_heads, qkv_bias=True,
                                       attn_drop=0., proj_drop=0., use_mask=use_mask)
        # self.intermediate = BertIntermediate(hidden_size, intermediate_size)
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.act_fn = nn.GELU()

        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.drop_path = DropPath(
            drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.gamma_1 = nn.Parameter(torch.zeros(
            (hidden_size)), requires_grad=True)
        self.gamma_2 = nn.Parameter(torch.zeros(
            (hidden_size)), requires_grad=True)

    def ffn_forward(self, x, q):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        q = self.linear1(q)
        q = self.act_fn(q)
        q = self.linear2(q)
        return x, q

    def forward(self, x, q, q_mask, H, W):

        def _inner_forward(x, q, q_mask):
            x_, q_ = self.self_attn(self.norm1(x), self.norm1(q), q_mask, H, W)
            x = x + self.gamma_1 * self.drop_path(x_)
            q = q + self.gamma_1 * self.drop_path(q_)
            x_, q_ = self.ffn_forward(self.norm2(x), self.norm2(q))
            x = x + self.gamma_2 * self.drop_path(x_)
            q = q + self.gamma_2 * self.drop_path(q_)
            return x, q

        if self.with_cp and x.requires_grad:
            x, q = cp.checkpoint(_inner_forward, x, q, q_mask)
        else:
            x, q = _inner_forward(x, q, q_mask)

        return x, q


class MultiModelBertEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 window_attn=False, window_size=14, with_cp=False, use_mask=False,
                 pretrained=None):

        super(MultiModelBertEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer

        window_attn = [window_attn] * \
            depth if not isinstance(window_attn, list) else window_attn
        window_size = [window_size] * \
            depth if not isinstance(window_size, list) else window_size
        logging.info('window attention:', window_attn)
        logging.info('window size:', window_size)
        logging.info('use mask:', use_mask)
        print('use mask:', use_mask)
        layers = []
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            layers.append(
                MultiModelBertLayer(hidden_size=embed_dim, intermediate_size=int(embed_dim * mlp_ratio),
                                    num_attention_heads=num_heads, drop_path_ratio=dpr[i],
                                    windowed=window_attn[i], window_size=window_size[i],
                                    with_cp=with_cp, use_mask=use_mask)
            )

        self.layers = nn.ModuleList(layers)
        self.visual_embed = VisualPatchEmbedding(in_dim=in_chans, out_dim=embed_dim,
                                                 patch_size=patch_size, image_size=img_size)
        self.token_embed = TokenBaseEmbedding(dim=embed_dim, vocab_size=49411)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu',
                            strict=False, logger=logger)

    def forward(self, img, question):
        x, H, W = self.visual_embed(img)
        q = self.token_embed(question)

        for layer in self.layers:
            x, q = layer(x, q, H, W)
        return x, q
