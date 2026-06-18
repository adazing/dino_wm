from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from .diffusion_ema import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch.nn as nn
import einops
import torch.nn.functional as F
import sys
from models.vq_behavior_transformer.gpt import generate_mask_matrix


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        x : [ batch_size x in_channels x horizon ]
        cond : [ batch_size x cond_dim]

        returns:
        out : [ batch_size x out_channels x horizon ]
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
        n_patches: int = 1,
    ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon

        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1

        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps * n_patches

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None

        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer, num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer, num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=n_layer
            )
        self.n_patches = n_patches
        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = (
                mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
            )
            self.register_buffer("mask", mask)

            # print(">>> DEBUG: mask (causal) shape:", mask.shape)  # should be (T, T)
            # print("mask[0,:5] =", mask[0, :5])   # first row, first 5 entries
            # print("mask[-1,:5] =", mask[-1, :5]) # last row, first 5 entries

            if time_as_cond and obs_as_cond:
                # Build patch-aware memory mask:
                # S = total cond tokens = 1 (time) + n_obs_steps * n_patches
                S_patches = n_obs_steps * self.n_patches
                S = 1 + S_patches

                # generate_mask_matrix returns shape (1,1,S_patches,S_patches)
                # squeeze to (S_patches, S_patches)
                patch_block = generate_mask_matrix(self.n_patches, n_obs_steps).squeeze(0).squeeze(0)
                # print(self.n_patches, n_obs_steps)
                # print(">>> DEBUG: patch_block (before sanity check) shape:", patch_block.shape)
                # sanity check
                assert patch_block.shape == (S_patches, S_patches), "patch_block shape mismatch"

                # convert to bool for logical ops
                patch_block = patch_block.bool()

                # create memory_mask boolean (T, S) where True => allowed
                mem_bool = torch.zeros((T, S), dtype=torch.bool)
                mem_bool[:, 0] = True  # allow decoder positions to always see the time token (s=0)

                # For decoder timestep t, allow patches from observation-windows up through obs_step.
                # Map decoder t -> obs window index (clamped).
                for t_idx in range(T):
                    obs_step = min(t_idx, n_obs_steps - 1)  # clamp to valid obs windows
                    # rows corresponding to windows 0..obs_step:
                    rows_to_use = (obs_step + 1) * self.n_patches  # count of patch-rows to include
                    # take those rows from patch_block and see which columns are reachable
                    rows = patch_block[:rows_to_use, :]               # shape (rows_to_use, S_patches)
                    allowed_cols = rows.any(dim=0)                    # shape (S_patches,)
                    # fill memory boolean skipping time token at index 0
                    mem_bool[t_idx, 1:] = allowed_cols

                # Convert boolean mask to additive-style mask expected by nn.Transformer: 0.0 allowed / -inf blocked
                mem_mask = mem_bool.float().masked_fill(~mem_bool, float("-inf")).masked_fill(mem_bool, float(0.0))
                self.register_buffer("memory_mask", mem_mask)

                # print(">>> DEBUG: patch_block shape:", patch_block.shape)  # (S_patches, S_patches)
                # print("patch_block (top-left 5x5):\n", patch_block[:5, :5].int())
                # print(">>> DEBUG: memory_mask shape:", mem_mask.shape)  # (T, S)
                # print("row 0 sum (tokens allowed at t=0):", mem_bool[0].sum().item())
                # print("row 1 sum (tokens allowed at t=1):", mem_bool[1].sum().item())
                # print("last row sum (tokens allowed at t=T-1):", mem_bool[-1].sum().item())
                # for t_idx in range(T):
                #     allowed = torch.where(mem_bool[t_idx])[0]
                #     print(f"t={t_idx}, allowed_count={len(allowed)}, first10={allowed[:10].tolist()}")
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                "in_proj_weight",
                "q_proj_weight",
                "k_proj_weight",
                "v_proj_weight",
            ]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)

            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:, 1:, :]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)

            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask
            )
            # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
    ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print(
            "number of parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        global_cond=None,
    ):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1, -2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1, -2)
        # (B,T,C)
        return x


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        goal_dim: int,
        obs_horizon: int,
        pred_horizon: int,
        action_horizon: int,
        views=1,
        data_act_scale=1.0,
        visual_input=False,
        p_drop_emb: float = 0.0,
        p_drop_attn: float = 0.1,
        n_layer: int = 8,
        n_head: int = 4,
        n_embd: int = 256,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        use_transform=None,
        n_patches=1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = act_dim
        self.goal_dim = goal_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.ema_noise_pred_net = None
        self.data_act_scale = data_act_scale
        self.visual_input = visual_input
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_transform = use_transform

        self.obs_dim = obs_dim if self.goal_dim == 0 else obs_dim + goal_dim

        if goal_dim == 0:
            self.cond_method = "unconditional"
        else:
            self.cond_method = "stack"
        # create network object
        ######################## jay cnn #########################
        # self.noise_pred_net = ConditionalUnet1D(
        #     input_dim=self.action_dim,
        #     global_cond_dim=self.obs_dim*self.obs_horizon
        # ).cuda()
        ##################### jay transformer #####################
        self.noise_pred_net = TransformerForDiffusion(
            input_dim=self.action_dim,
            output_dim=self.action_dim,
            horizon=pred_horizon,
            n_obs_steps=obs_horizon,
            cond_dim=self.obs_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_embd,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            n_cond_layers=0,
            n_patches=n_patches * views,
        ).cuda()
        #############################################################
        # for this demo, we use DDPMScheduler with 100 diffusion iterations
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        self.ema = EMAModel(
            self.noise_pred_net,
            inv_gamma=1.0,
            max_value=0.9999,
            min_value=0.0,
            power=0.75,
            update_after_step=0,
        )


    def forward(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        if action_seq is None:
            return self._predict(obs_seq, goal_seq, action_seq)
        else:
            pred_action, _, _ = self._predict(obs_seq, goal_seq, action_seq)
            _, loss, loss_dict = self._update(obs_seq, goal_seq, action_seq)
            return pred_action, loss, loss_dict

    def _update(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        # Assume dimensions are N T D for N sequences of T timesteps with dimension D.
        device = self.noise_pred_net.device

        # --- Build obs_cond (B, T_obs * n_patches, patch_dim) ---
        if obs_seq.ndim == 4:
            B, T_obs, n_patches, patch_dim = obs_seq.shape
        elif obs_seq.ndim == 3:
            obs_seq = obs_seq.unsqueeze(2)  # add n_patches dim
            B, T_obs, n_patches, patch_dim = obs_seq.shape
        else:
            raise ValueError(f"Unexpected obs_seq.ndim {obs_seq.ndim}")

        assert T_obs <= self.obs_horizon, f"T_obs {T_obs} > obs_horizon {self.obs_horizon}!!!!!"

        # pad time windows if needed (repeat first time-step's patch block)
        if T_obs < self.obs_horizon:
            missing_time = self.obs_horizon - T_obs
            pad_tokens = obs_seq[:, 0:1].repeat(1, missing_time, 1, 1)
            obs_seq = torch.cat([pad_tokens, obs_seq], dim=1)

        # handle goal stacking here
        if self.cond_method == "unconditional":
            cond = obs_seq
        elif self.cond_method == "stack":
            cond = torch.cat([goal_seq, obs_seq], dim=-1)

        obs_patches = einops.rearrange(cond, 'b t p d -> b (t p) d')
        obs_cond = obs_patches.to(device)
        obs_cond = obs_cond.to(next(self.noise_pred_net.parameters()).dtype)

        # --- Prepare actions and noise ---
        assert action_seq is not None
        naction = self.normalize_data(action_seq).to(device).to(next(self.noise_pred_net.parameters()).dtype)
        B = naction.shape[0]
        noise = torch.randn_like(naction, device=device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

        # predict noise (pass the full obs_cond)
        noise_pred = self.noise_pred_net(noisy_actions, timesteps, cond=obs_cond)

        loss = nn.functional.mse_loss(noise_pred, noise)
        loss_dict = {"total_loss": loss.detach().cpu().item()}
        return None, loss, loss_dict

    def normalize_data(self, data):
        return self.normalizer.normalize(data)

    def unnormalize_data(self, data):
        return self.normalizer.unnormalize(data)

    def _predict(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:

        device = self.noise_pred_net.device

        if obs_seq.ndim == 4:
            B, T_obs, n_patches, patch_dim = obs_seq.shape
        elif obs_seq.ndim == 3:
            obs_seq = obs_seq.unsqueeze(2)  # add n_patches dim
            B, T_obs, n_patches, patch_dim = obs_seq.shape
        else:
            raise ValueError(f"Unexpected obs_seq.ndim {obs_seq.ndim}")

        assert T_obs <= self.obs_horizon, f"T_obs {T_obs} > obs_horizon {self.obs_horizon}!!!!!"

        # pad time windows if needed (repeat first time-step's patch block)
        if T_obs < self.obs_horizon:
            missing_time = self.obs_horizon - T_obs
            pad_tokens = obs_seq[:, 0:1].repeat(1, missing_time, 1, 1)
            obs_seq = torch.cat([pad_tokens, obs_seq], dim=1)

        # handle goal stacking here
        if self.cond_method == "unconditional":
            cond = obs_seq
        elif self.cond_method == "stack":
            cond = torch.cat([goal_seq, obs_seq], dim=-1)

        obs_patches = einops.rearrange(cond, 'b t p d -> b (t p) d')
        obs_cond = obs_patches.to(device)
        obs_cond = obs_cond.to(next(self.noise_pred_net.parameters()).dtype)

        # sampling with EMA model
        self.ema_noise_pred_net = self.ema.averaged_model
        B = obs_cond.shape[0]
        noisy_action = torch.randn((B, self.pred_horizon, self.action_dim), device=device, dtype=next(self.noise_pred_net.parameters()).dtype)
        naction = noisy_action

        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.ema_noise_pred_net(sample=naction, timestep=k, cond=obs_cond)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

        naction = naction.detach()
        action_pred = self.unnormalize_data(naction)
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[:, start:end, :]
        # action = action_pred[:, :self.action_horizon, :]
        return action, None, None

    def ema_step(self):
        self.ema.step(self.noise_pred_net)

    def get_ema_average(self):
        return self.ema.averaged_model

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optimizer = torch.optim.AdamW(
            params=self.noise_pred_net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        print("lr:", self.lr)
        print("weight_decay:", self.weight_decay)
        return optimizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def train(self, mode=True):
        super().train(mode)
