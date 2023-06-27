import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.hub
from functools import partial
import numpy as np

from mmcv.runner import BaseModule
from ..builder import FUSIONS

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):

        B, N, C = x.shape
        q = self.wq(x[:, 0:int(N/2), ...]).reshape(B, int(N/2), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.wk(x[:, (int(N/2)):, ...]).reshape(B, int(N/2), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.wv(x[:, (int(N/2)):, ...]).reshape(B, int(N/2), self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, int(N/2), C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Para_Att(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm, has_mlp=False, activation="relu"):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.self_attn_x = SelfAttention(dim, num_heads=num_heads,
                                       qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.has_mlp = has_mlp

    def forward(self, x):
        B, N, C = x.shape
        x_CA = self.cross_attn(self.norm1(x))
        x_SA = self.self_attn_x(x[:, int(N / 2):, ...])

        return x_CA, x_SA

class MLP(nn.Module):

    def __init__(self, input_dim, inter_dim=None, output_dim=None, activation="relu", drop=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.output_dim = output_dim
        if inter_dim is None: self.inter_dim=input_dim
        if output_dim is None: self.output_dim=input_dim

        self.linear1 = nn.Linear(self.input_dim, self.inter_dim)
        self.activation = self._get_activation_fn(activation)
        self.dropout3 = nn.Dropout(drop)
        self.linear2 = nn.Linear(self.inter_dim, self.output_dim)
        self.dropout4 = nn.Dropout(drop)
        self.norm3 = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        x = self.linear2(self.dropout3(self.activation(self.linear1(x))))
        x = x + self.dropout4(x)
        x = self.norm3(x)
        return x

    def _get_activation_fn(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

@ FUSIONS.register_module()
class AMF(BaseModule):
    def __init__(self, in_channels, num_heads, out_channels=None, fusion_loss=None, lamda = 1, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.criterion = fusion_loss if fusion_loss is not None else nn.MSELoss(reduction='mean')
        self.lamda = lamda

        for idx, (in_c, n_h) in enumerate(zip(self.in_channels, self.num_heads)):


            setattr(self, 'Para_Att_s_' + str(idx), Para_Att(dim=in_c, num_heads=n_h))
            setattr(self, 'Para_Att_t_' + str(idx), Para_Att(dim=in_c, num_heads=n_h))
            setattr(self, 'mlp_' + str(idx), MLP(input_dim=in_c*3, output_dim=in_c))

    def forward(self, x_s, x_t):
        outs = []
        for idx in range(len(self.in_channels)):
            hwc_shape = [x_s[idx].shape[2], x_s[idx].shape[3], x_s[idx].shape[1]]
            # x_s_i = x_s[idx]
            # x_t_i = x_t[idx]

            x_s_i_token = x_s[idx].view(-1, hwc_shape[2], hwc_shape[0]*hwc_shape[1]).permute(0, 2, 1)
            x_t_i_token = x_t[idx].view(-1, hwc_shape[2], hwc_shape[0]*hwc_shape[1]).permute(0, 2, 1)

            x_s_i_tmp = torch.cat((x_t_i_token, x_s_i_token), dim=1)
            x_t_i_tmp = torch.cat((x_s_i_token, x_t_i_token), dim=1)

            x_s_CA, x_s_SA = getattr(self, 'Para_Att_s_' + str(idx))(x_s_i_tmp)
            x_t_CA, x_t_SA = getattr(self, 'Para_Att_t_' + str(idx))(x_t_i_tmp)

            x_s_SA = x_s_SA + x_s_i_token
            x_t_SA = x_t_SA + x_t_i_token

            x_inter = (x_s_CA + x_t_CA).permute(0, 2, 1).view(-1, hwc_shape[2], hwc_shape[0], hwc_shape[1])
            x_s_SA = x_s_SA.permute(0, 2, 1).view(-1, hwc_shape[2], hwc_shape[0], hwc_shape[1])
            x_t_SA = x_t_SA.permute(0, 2, 1).view(-1, hwc_shape[2], hwc_shape[0], hwc_shape[1])
            out = torch.concat((x_s_SA, x_inter, x_t_SA), dim=1)

            out_ = out.view(-1, self.in_channels[idx]*3, hwc_shape[0]*hwc_shape[1]).permute(0, 2, 1)
            out = getattr(self, 'mlp_' + str(idx))(out_)
            out = out.permute(0, 2, 1).view(-1, self.in_channels[idx], hwc_shape[0], hwc_shape[1])

            outs.append(out)

        return tuple(outs)

    def fusion_loss(self):
        return None
