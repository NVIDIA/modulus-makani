import math

import torch.nn.functional as F
import torch
import torch.nn as nn
from functools import partial
from makani.networks.layers import DropPath

# mp stuff
from makani.utils import comm
from makani.networks.layers import MLP, PatchEmbed
from makani.mpu.layers import DistributedMatmul, DistributedMLP, DistributedAttention


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            input_format="traditional",
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop_rate=0.,
            proj_drop_rate=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(dim, dim)

        if proj_drop_rate > 0:
            self.proj_drop = nn.Dropout(proj_drop_rate)
        else:
            self.proj_drop = nn.Identity()


    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_rate)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 mlp_drop_rate=0., attn_drop_rate=0., path_drop_rate=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 comm_inp_name="fin", comm_hidden_name="fout"):
        super().__init__()

        if (comm.get_size(comm_inp_name) * comm.get_size(comm_hidden_name)) > 1:
            self.attn = DistributedAttention(
                dim, input_format="traditional",
                comm_inp_name=comm_inp_name, comm_hidden_name=comm_hidden_name,
                num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate, proj_drop_rate=mlp_drop_rate,
                norm_layer=norm_layer)
        else:
            self.attn = Attention(
                dim, input_format="traditional",
                num_heads=num_heads, qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate, proj_drop_rate=mlp_drop_rate,
                norm_layer=norm_layer)
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0. else nn.Identity()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        # distribute MLP for model parallelism
        if (comm.get_size(comm_inp_name) * comm.get_size(comm_hidden_name)) > 1:
            self.mlp = DistributedMLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                                      act_layer=act_layer, drop_rate=mlp_drop_rate,
                                      input_format="traditional",
                                      comm_inp_name=comm_inp_name,
                                      comm_hidden_name=comm_hidden_name
                                      )
        else:
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                           act_layer=act_layer, drop_rate=mlp_drop_rate, input_format="traditional")

    def forward(self, x):

        # flatten transpose:
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = self.norm2(x)
        x = x + self.drop_path(self.mlp(x))

        return x

class VisionTransformer(nn.Module):
    def __init__(self, inp_shape=[224, 224], patch_size=(16, 16),
                 inp_chans=3, out_chans=3, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False,
                 mlp_drop_rate=0., attn_drop_rate=0., path_drop_rate=0.,
                 norm_layer="layer_norm", comm_inp_name="fin", comm_hidden_name="fout", **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = inp_shape
        self.out_ch = out_chans
        self.comm_inp_name = comm_inp_name
        self.comm_hidden_name = comm_hidden_name

        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=patch_size, in_chans=inp_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=path_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, depth)]  # stochastic depth decay rule

        if norm_layer == "layer_norm":
            norm_layer_handle = nn.LayerNorm
        else:
            raise NotImplementedError(f"Error, normalization layer type {norm_layer} not implemented for ViT.")

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                mlp_drop_rate=mlp_drop_rate, attn_drop_rate=attn_drop_rate, path_drop_rate=dpr[i],
                norm_layer=norm_layer_handle, comm_inp_name=comm_inp_name, comm_hidden_name=comm_hidden_name)
            for i in range(depth)])

        self.norm = norm_layer_handle(embed_dim)

        self.out_size = self.out_ch * self.patch_size[0] * self.patch_size[1]

        self.head = nn.Linear(embed_dim, self.out_size, bias=False)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x).transpose(1,2)  # patch linear embedding

        # add positional encoding to each token
        x = x + self.pos_embed
        return self.pos_drop(x)

    def forward_head(self, x):
        B, _, _ = x.shape # B x N x embed_dim
        x = x.reshape(B, self.patch_embed.red_img_size[0], self.patch_embed.red_img_size[1], self.embed_dim)
        B, h, w, _ = x.shape

        # apply head
        x = self.head(x)
        x = x.reshape(shape=(B, h, w, self.patch_size[0], self.patch_size[1], self.out_ch))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_ch, self.img_size[0], self.img_size[1]))

        return x

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = self.forward_head(x)
        return x
