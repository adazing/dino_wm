import torch
import einops
import torch.nn as nn
from transformers import AutoModel
from torchvision import transforms

class DinoV3Encoder(nn.Module):
    def __init__(self, name, feature_key, plus=False, output_dim=None, postprocess=None, n_patches=196):
        super().__init__()
        print("Encoder feature_key:", feature_key)
        self.name = name
        if plus:
            model_name = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
        else:
            model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        self.base_model = AutoModel.from_pretrained(
            model_name,
            # device_map="auto",
            # use_auth_token=True,
            trust_remote_code=True,
            )
        self.feature_key = feature_key
        self.emb_dim = self.base_model.config.hidden_size
        self.output_dim = self.emb_dim # for compatibility
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.config.patch_size

        # TODO: sanity check
        self.postprocess = postprocess
        if postprocess is not None:
            if postprocess == 'avg_pool':
                self.latent_ndim = 1

        self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # self.normalization = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def forward(self, x):
        # Accept arbitrary number of leading dimensions before (C, H, W)
        # and preserve them on return.
        # Example: input shape (...prefix, C, H, W)
        assert x.max() <= 1.0 + 1e-4 and x.min() >= -1e-4, "expect [0,1] range"
        x = self.normalization(x)

        prefix_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]

        # Collapse all leading dims into a single batch dimension for the base model
        prod_prefix = 1
        for d in prefix_shape:
            prod_prefix *= d

        x = x.reshape(prod_prefix, c, h, w)

        outputs = self.base_model(pixel_values=x)
        if self.feature_key == "x_norm_clstoken":
            emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif self.feature_key == "x_norm_patchtokens":
            emb = outputs.last_hidden_state[:, 5:, :]  # Patch tokens (skip 4 register tokens)

        emb = emb.reshape(*prefix_shape, *emb.shape[1:])

        if self.postprocess == 'avg_pool':
            emb = torch.mean(emb, dim=-2) # (...prefix, e)

        if self.latent_ndim == 1:
            emb = emb.unsqueeze(len(prefix_shape)) # dummy patch dim, b v 1 e
        return emb
