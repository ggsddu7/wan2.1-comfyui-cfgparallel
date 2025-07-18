# original version: https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import time

import math

import torch
import torch.nn as nn
from einops import repeat

from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.flux.layers import EmbedND
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.diffusionmodules.mmdit import RMSNorm
import comfy.ldm.common_dit
import comfy.model_management

import torch.distributed as dist

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float32)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6, operation_settings={}):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.k = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.o = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.norm_q = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

        spg = None # dist.new_group()
        from deepspeed.sequence.layer import DistributedAttention
        self.dist_attn = DistributedAttention(optimized_attention, spg)


    def forward(self, x, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n * d)
            return q, k, v

        q, k, v = qkv_fn(x)
        rank, dist_inited = -1, dist.is_initialized()
        if dist_inited:
            rank = dist.get_rank()
        print(f"WanSelfAttention-00 {rank} {x.shape} ==> {q.shape} {k.shape} {v.shape}")
        q, k = apply_rope(q, k, freqs)
        if not dist_inited:
            x = optimized_attention(
                q.view(b, s, n * d),
                k.view(b, s, n * d),
                v,
                heads=self.num_heads,
            )
        else:
            # q = q.view(b, s, n * d)
            # k = k.view(b, s, n * d)
            v = v.view(b, s, n, d)

            x = self.dist_attn(q, k, v, 0, heads=self.num_heads)
        """
        if False:
            x = optimized_attention(
                q.view(b, s, n * d),
                k.view(b, s, n * d),
                v,
                heads=self.num_heads,
            )
        else:
            aa = (torch.arange(8).reshape(1,4,2)+8*rank).to("cuda")
            a2aout = torch.empty_like(aa)
            dist.all_to_all_single(a2aout, aa)
            print(f"{rank} {torch.arange(16).reshape(1,8,2)} | {aa} | {a2aout} | {a2aout.reshape(1,8,1)}")
            print(f"{rank} {a2aout.reshape(1,8,1).reshape(1,4,2)}")
            xxxx
            print(f"WanSelfAttention-11 {rank} {x.shape} ==> {q.shape} {k.shape} {v.shape}")
            P = torch.cuda.device_count()
            s_ = s // P
            num_heads = n // P

            q = q[:,rank*s_:(rank+1)*s_,:]
            a2aout = torch.empty_like(q)
            print(f"WanSelfAttention-11 {rank} {x.shape} ==> {q.shape} {a2aout.shape}")
            dist.all_to_all_single(a2aout, q)
            q = a2aout.reshape(b, s, n*d//P)
            print(f"WanSelfAttention-22 {rank} {x.shape} ==> {q.shape} {a2aout.shape}")

            k = k[:,rank*s_:(rank+1)*s_,:]
            a2aout = torch.empty_like(k)
            print(f"WanSelfAttention-33 {rank} {x.shape} ==> {k.shape} {a2aout.shape}")
            dist.all_to_all_single(a2aout, k)
            k = a2aout.reshape(b, s, n*d//P)
            print(f"WanSelfAttention-44 {rank} {x.shape} ==> {k.shape} {a2aout.shape}")

            v = v[:,rank*s_:(rank+1)*s_,:]
            a2aout = torch.empty_like(v)
            print(f"WanSelfAttention-55 {rank} {x.shape} ==> {v.shape} {a2aout.shape}")
            dist.all_to_all_single(a2aout, v)
            v = a2aout.reshape(b, s, n*d//P)
            print(f"WanSelfAttention-66 {rank} {x.shape} ==> {v.shape} {a2aout.shape}")

            x = optimized_attention(q, k, v, heads=num_heads).reshape(b, s_, n * d)
            print(f"WanSelfAttention-77 {rank} {x.shape}")
            xxxxx
        """

        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        # compute query, key, value
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(context))
        v = self.v(context)

        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads)

        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6, operation_settings={}):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, operation_settings=operation_settings)

        self.k_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.v_img = operation_settings.get("operations").Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = RMSNorm(dim, eps=eps, elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if qk_norm else nn.Identity()

    def forward(self, x, context):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
        """
        t1 = time.time() * 1000
        context_img = context[:, :257]
        context = context[:, 257:]
        t2 = time.time() * 1000

        # compute query, key, value
        q = self.norm_q(self.q(x))
        t3 = time.time() * 1000
        k = self.norm_k(self.k(context))
        t4 = time.time() * 1000
        v = self.v(context)
        t5 = time.time() * 1000
        k_img = self.norm_k_img(self.k_img(context_img))
        t6 = time.time() * 1000
        v_img = self.v_img(context_img)
        t7 = time.time() * 1000
        img_x = optimized_attention(q, k_img, v_img, heads=self.num_heads)
        t8 = time.time() * 1000
        # compute attention
        x = optimized_attention(q, k, v, heads=self.num_heads)
        t9 = time.time() * 1000

        # output
        x = x + img_x
        t10 = time.time() * 1000
        x = self.o(x)
        t11 = time.time() * 1000
        # print(f"WanI2VCrossAttention {t2-t1:.0f} {t3-t2:.0f} {t4-t3:.0f} {t5-t4:.0f} {t6-t5:.0f} {t7-t6:.0f} {t8-t7:.0f} {t9-t8:.0f} {t10-t9:.0f} {t11-t10:.0f} {t11-t1:.0f}")
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, operation_settings=operation_settings)
        self.norm3 = operation_settings.get("operations").LayerNorm(
            dim, eps,
            elementwise_affine=True, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps, operation_settings=operation_settings)
        self.norm2 = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.ffn = nn.Sequential(
            operation_settings.get("operations").Linear(dim, ffn_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operation_settings.get("operations").Linear(ffn_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 6, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(
        self,
        x,
        e,
        freqs,
        context,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32

        t0 = time.time() * 1000
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e).chunk(6, dim=1)
        # assert e[0].dtype == torch.float32

        # self-attention
        t1 = time.time() * 1000
        """
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0],
            freqs)
        """
        aa = self.norm1(x)
        ta = time.time() * 1000
        bb = aa * (1 + e[1]) + e[0]
        tb = time.time() * 1000
        y = self.self_attn(bb, freqs)
        # torch.cuda.synchronize() # a2a_qkvx需要及时释放现存
        t2 = time.time() * 1000

        x = x + y * e[2]
        t3 = time.time() * 1000

        # cross-attention & ffn
        # x = x + self.cross_attn(self.norm3(x), context)
        cc = self.norm3(x)
        tc = time.time() * 1000
        dd = self.cross_attn(cc, context)
        td = time.time() * 1000
        x = x + dd

        t4 = time.time() * 1000
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        t5 = time.time() * 1000
        x = x + y * e[5]
        t6 = time.time() * 1000
        # print(f"WanAttentionBlock {self.bidx} {t1-t0:.0f} {t2-t1:.0f} {t3-t2:.0f} {t4-t3:.0f} {t5-t4:.0f} {t6-t5:.0f} {t6-t0:.0f}", end=" | ")
        # print(f"WanAttentionBlock {self.bidx} {t1-t0:.0f} {t2-t1:.0f}({ta-t1:.0f} {tb-ta:.0f} {t2-tb:.0f}) {t3-t2:.0f} {t4-t3:.0f}({tc-t3:.0f} {td-tc:.0f} {t4-td:.0f}) {t5-t4:.0f} {t6-t5:.0f} {t6-t0:.0f}")
        # print("-----------------------------------------------------------------------------------")
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, operation_settings={}):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = operation_settings.get("operations").LayerNorm(dim, eps, elementwise_affine=False, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))
        self.head = operation_settings.get("operations").Linear(dim, out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype"))

        # modulation
        self.modulation = nn.Parameter(torch.empty(1, 2, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        # assert e.dtype == torch.float32
        e = (comfy.model_management.cast_to(self.modulation, dtype=x.dtype, device=x.device) + e.unsqueeze(1)).chunk(2, dim=1)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, operation_settings={}):
        super().__init__()

        self.proj = torch.nn.Sequential(
            operation_settings.get("operations").LayerNorm(in_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), operation_settings.get("operations").Linear(in_dim, in_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            torch.nn.GELU(), operation_settings.get("operations").Linear(in_dim, out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")),
            operation_settings.get("operations").LayerNorm(out_dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(torch.nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 image_model=None,
                 device=None,
                 dtype=None,
                 operations=None,
                 ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()
        self.dtype = dtype
        operation_settings = {"operations": operations, "device": device, "dtype": dtype}

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = operations.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size, device=operation_settings.get("device"), dtype=torch.float32)
        self.text_embedding = nn.Sequential(
            operations.Linear(text_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.GELU(approximate='tanh'),
            operations.Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        self.time_embedding = nn.Sequential(
            operations.Linear(freq_dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")), nn.SiLU(), operations.Linear(dim, dim, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))
        self.time_projection = nn.Sequential(nn.SiLU(), operations.Linear(dim, dim * 6, device=operation_settings.get("device"), dtype=operation_settings.get("dtype")))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps, operation_settings=operation_settings)
            for _ in range(num_layers)
        ])
        for bidx, blk in enumerate(self.blocks):
            blk.bidx = bidx

        # head
        self.head = Head(dim, out_dim, patch_size, eps, operation_settings=operation_settings)

        d = dim // num_heads
        self.rope_embedder = EmbedND(dim=d, theta=10000.0, axes_dim=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)])

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim, operation_settings=operation_settings)
        else:
            self.img_emb = None

    def forward_orig(
        self,
        x,
        t,
        context,
        clip_fea=None,
        freqs=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (Tensor):
                List of input video tensors with shape [B, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [B, L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        t1 = time.time() * 1000
        # embeddings
        # noise, repeat4(mask), concat_latent_image [1, 36, 17, 112, 64]
        x = self.patch_embedding(x.float()).to(x.dtype)
        t2 = time.time() * 1000
        grid_sizes = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        t3 = time.time() * 1000

        # time embeddings
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).to(dtype=x[0].dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        t4 = time.time() * 1000

        # context = cross_attn = text_encoder(prompt)
        context = self.text_embedding(context)

        if clip_fea is not None and self.img_emb is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
        # context = clip_vision_output.penultimate_hidden_states, cross_attn

        print("WanModel-00000", e0.shape, context_clip.shape, context.shape, grid_sizes, freqs.shape, x.shape)
        # import pudb; pu.db
        # arguments
        kwargs = dict(
            e=e0,
            freqs=freqs,
            context=context)

        """
        512*896*81=>[1, 16, 21, 112, 64]=>[1, 37632, 5120] ~22m
        512*896*73=>[1, 16, 19, 112, 64]=>[1, 34048, 5120] # 分层加载临界点
        512*896*65=>[1, 16, 17, 112, 64]=>[1, 30464, 5120]
        512*896*49=>[1, 16, 13, 112, 64]=>[1, 23296, 5120]
        512*896*33=>[1, 16,  9, 112, 64]=>[1, 16128, 5120]
        512*896*17=>[1, 16,  5, 112, 64]=>[1,  8960, 5120]

        720*1280*81=>[1, 16, 21, 160, 90]=>[1, 75600, 5120] ~70m
        720*1280*65=>[1, 16, 17, 160, 90]=>[1, 61200, 5120]
        720*1280*49=>[1, 16, 13, 160, 90]=>[1, 46800, 5120]
        720*1280*33=>[1, 16,  9, 160, 90]=>[1, 32400, 5120]
        720*1280*17=>[1, 16,  5, 160, 90]=>[1, 18000, 5120]
        """
        xs1, nblock = x.shape[1], len(self.blocks)
        print("XXXXXX", xs1, nblock)
        if True: # xs1 <= 34048: # 不分层加载
            t5 = time.time() * 1000
            ### dist 分x和freqs
            dist_inited = dist.is_initialized()
            if dist_inited:
                rank = dist.get_rank()
                world_size = torch.cuda.device_count()
                shared_seqlen = x.shape[1] // world_size
                print(f"==chunk0== {rank} {x.shape} {shared_seqlen} {kwargs['freqs'].shape}")
                x = x[:,rank*shared_seqlen:(rank+1)*shared_seqlen,:]
                kwargs['freqs'] = kwargs['freqs'][:,rank*shared_seqlen:(rank+1)*shared_seqlen,:,:,:,:]
                print(f"==chunk1== {rank} {x.shape} {shared_seqlen} {kwargs['freqs'].shape}")
                for block in self.blocks:
                    x = block(x, **kwargs)
                torch.cuda.synchronize()
                # x: [1, 15232, 5120]
                output_list = [torch.zeros((1, shared_seqlen, 5120), dtype=x.dtype, device=x.device) for _ in range(world_size)]
                dist.all_gather(output_list, x.contiguous())
                x = torch.cat(output_list, dim=1)
            else:
                for block in self.blocks:
                    x = block(x, **kwargs)
                # x: [1, 30464, 5120]
            t6 = time.time() * 1000
        else: # 分层加载
            print("------------fencengjiazai------------")
            skconfig={75600: (2,4), 61200: (2, 20), 46800: (2, 30)}
            slen, klen = skconfig[min(filter(lambda k: k-xs1>=0, skconfig.keys()))]
            # print(f"### == {','.join([blk.cross_attn.q.weight.device.type for blk in self.blocks])} {xs1} {slen} {klen}")
            for bidx in range(0, klen):
                if self.blocks[bidx].cross_attn.q.weight.device.type == "cpu":
                    self.blocks[bidx].to("cuda")
            for bidx in range(klen, nblock):
                if self.blocks[bidx].cross_attn.q.weight.device.type == "cuda":
                    self.blocks[bidx].to("cpu", non_blocking=True)
            for bidx, block in enumerate(self.blocks):
                # print(f"### {bidx} {','.join([blk.cross_attn.q.weight.device.type for blk in self.blocks])}")
                for kk in range(bidx+slen, bidx, -1):
                    if kk >= klen and kk < nblock:
                        if self.blocks[kk].cross_attn.q.weight.device.type == "cpu":
                            self.blocks[kk].to("cuda", non_blocking=True)

                if block.cross_attn.q.weight.device.type == "cpu":
                    block.to("cuda")
                x = block(x, **kwargs)
                if bidx >= klen and bidx < nblock:
                    block.to("cpu", non_blocking=True)

        print("aaaa", x.shape, e.shape)
        # head
        x = self.head(x, e)
        print("bbbb", x.shape)
        t7 = time.time() * 1000

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        print("cccc", x.shape, grid_sizes)
        t8 = time.time() * 1000
        # print(f"WanModel {t2-t1:.0f} {t3-t2:.0f} {t4-t3:.0f} {t5-t4:.0f} {t6-t5:.0f} {t7-t6:.0f} {t8-t7:.0f} {t8-t1:.0f}")
        return x

    def forward(self, x, timestep, context, clip_fea=None, **kwargs):
        bs, c, t, h, w = x.shape
        x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
        patch_size = self.patch_size
        t_len = ((t + (patch_size[0] // 2)) // patch_size[0])
        h_len = ((h + (patch_size[1] // 2)) // patch_size[1])
        w_len = ((w + (patch_size[2] // 2)) // patch_size[2])
        img_ids = torch.zeros((t_len, h_len, w_len, 3), device=x.device, dtype=x.dtype)
        img_ids[:, :, :, 0] = img_ids[:, :, :, 0] + torch.linspace(0, t_len - 1, steps=t_len, device=x.device, dtype=x.dtype).reshape(-1, 1, 1)
        img_ids[:, :, :, 1] = img_ids[:, :, :, 1] + torch.linspace(0, h_len - 1, steps=h_len, device=x.device, dtype=x.dtype).reshape(1, -1, 1)
        img_ids[:, :, :, 2] = img_ids[:, :, :, 2] + torch.linspace(0, w_len - 1, steps=w_len, device=x.device, dtype=x.dtype).reshape(1, 1, -1)
        img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

        freqs = self.rope_embedder(img_ids).movedim(1, 2)
        return self.forward_orig(x, timestep, context, clip_fea=clip_fea, freqs=freqs)[:, :, :t, :h, :w]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [L, C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        u = x
        b = u.shape[0]
        u = u[:, :math.prod(grid_sizes)].view(b, *grid_sizes, *self.patch_size, c)
        u = torch.einsum('bfhwpqrc->bcfphqwr', u)
        u = u.reshape(b, c, *[i * j for i, j in zip(grid_sizes, self.patch_size)])
        return u
