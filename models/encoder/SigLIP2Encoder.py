import torch
import torch.nn as nn
import einops

from transformers import (
    AutoImageProcessor,
    AutoModel
)

class SigLIP2Encoder(nn.Module):
    def __init__(self, name, feature_key, output_dim=None, postprocess=None, n_patches=196):
        super().__init__()
        print("Encoder feature_key:", feature_key)
        self.name = name
        repo = "google/siglip2-base-patch16-224"
        self.processor = AutoImageProcessor.from_pretrained(
            repo,
            trust_remote_code=True,
        )

        # loads the multimodal SigLIP2 model (text + vision)
        self.full_model = AutoModel.from_pretrained(
            repo,
            trust_remote_code=True,
        )

        # extract the vision submodule
        self.base_model = self.full_model.vision_model
        self.feature_key = feature_key
        self.emb_dim = self.base_model.config.hidden_size
        self.output_dim = self.emb_dim # for compatibility

        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
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

        inputs = self.processor(
            images=x,
            return_tensors="pt",
            do_rescale=False,
        )

        device = next(self.base_model.parameters()).device
        target_dtype = next(self.base_model.parameters()).dtype
        inputs = {
            k: v.to(device=device, dtype=target_dtype) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Run just the vision module
        out = self.base_model(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
        )
        hidden = out.last_hidden_state
        emb = hidden.reshape(*prefix_shape, *hidden.shape[1:])

        if self.postprocess == "avg_pool":
            emb = emb.mean(dim=-2) # (...prefix, e)

        if self.latent_ndim == 1:
            emb = emb.unsqueeze(len(prefix_shape))
        return emb
