import torch
import einops
import torch.nn as nn
from transformers import AutoModel, AutoVideoProcessor
import einops
from transformers.video_utils import VideoMetadata

class VJEPA2Encoder(nn.Module):
    def __init__(self, name, feature_key, output_dim=None, postprocess=None, n_patches=256):
        super().__init__()
        print("Encoder feature_key:", feature_key)
        self.name = name
        repo = "facebook/vjepa2-vitl-fpc64-256"
        self.base_model = AutoModel.from_pretrained(
            repo,
            # use_auth_token=True,
            trust_remote_code=True,
            )
        self.processor = AutoVideoProcessor.from_pretrained(
            repo,
            trust_remote_code=True,
            )
        self.feature_key = feature_key
        self.emb_dim = self.base_model.config.hidden_size
        self.output_dim = self.emb_dim # for compatibility
        self.latent_ndim = 2
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
        inputs = self.processor(x, return_tensors="pt", do_rescale=False)["pixel_values_videos"]
        T = 2 # TODO: might have to change based on how many frames the model expects
        video_batch = einops.rearrange(inputs, "t b c h w -> b t c h w").repeat(1, T, 1, 1, 1)
        outputs = self.base_model.get_vision_features(video_batch)
        emb = outputs.reshape(*prefix_shape, *outputs.shape[1:])

        if self.postprocess == 'avg_pool':
            emb = torch.mean(emb, dim=-2) # (...prefix, e)

        if self.latent_ndim == 1:
            emb = emb.unsqueeze(len(prefix_shape)) # dummy patch dim, b v 1 e
        return emb
