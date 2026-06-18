import torch
import einops
import torch.nn as nn
from transformers import AutoImageProcessor, Dinov2Model


class WebSSLEncoder(nn.Module):
    def __init__(self, name, feature_key, plus=False, output_dim=None, postprocess=None, n_patches=196):
        super().__init__()
        print("Encoder feature_key:", feature_key)
        self.name = name
        repo = "facebook/webssl-dino300m-full2b-224"
        self.processor = AutoImageProcessor.from_pretrained(repo, do_rescale=False)
        self.base_model = Dinov2Model.from_pretrained(repo)

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

    def forward(self, x):
        # Accept arbitrary number of leading dimensions before (C, H, W)
        # and preserve them on return.
        # Example: input shape (...prefix, C, H, W)
        prefix_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]

        # Collapse all leading dims into a single batch dimension for the base model
        prod_prefix = 1
        for d in prefix_shape:
            prod_prefix *= d

        x = x.reshape(prod_prefix, c, h, w)

        inputs = self.processor(images=x, return_tensors="pt")
        
        device = next(self.base_model.parameters()).device
        target_dtype = next(self.base_model.parameters()).dtype
        inputs = {
            k: v.to(device=device, dtype=target_dtype) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        outputs = self.base_model(**inputs)
        if self.feature_key == "x_norm_clstoken":
            emb = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif self.feature_key == "x_norm_patchtokens":
            emb = outputs.last_hidden_state[:, 1:, :]  # Patch tokens (skip 4 register tokens)
        
        emb = emb.reshape(*prefix_shape, *emb.shape[1:])

        if self.postprocess == 'avg_pool':
            emb = torch.mean(emb, dim=-2) # (...prefix, e)

        if self.latent_ndim == 1:
            emb = emb.unsqueeze(len(prefix_shape)) # dummy patch dim, b v 1 e
        return emb