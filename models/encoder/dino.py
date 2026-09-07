import torch
import einops
import torch.nn as nn
from torchvision import transforms

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True

class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key, output_dim=None, postprocess=None, n_patches=256):
        super().__init__()
        print("Encoder feature_key:", feature_key)
        self.name = name
        self.base_model = torch.hub.load("facebookresearch/dinov2:b48308a", name)
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        self.output_dim = self.emb_dim # for compatibility
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size
        self.n_patches = n_patches

        # TODO: sanity check
        self.postprocess = postprocess
        if postprocess is not None:
            if postprocess == 'avg_pool':
                self.latent_ndim = 1

        # # DINOv2 expects ImageNet-normalized inputs derived from pixels in [0, 1].
        # self.normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Original dino_wm fed DINO images in [-1, 1] (default_transform's
        # Normalize(0.5, 0.5)), NOT ImageNet-normalized. Reproduce that here so
        # existing WM checkpoints work without retraining, while default_transform
        # stays in [0, 1] for the patch encoders. (x - 0.5) / 0.5 == 2x - 1.
        self.normalization = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def forward(self, x):
        # Accept arbitrary number of leading dimensions before (C, H, W)
        # and preserve them on return.
        # Example: input shape (...prefix, C, H, W)
        assert x.max() <= 1.0 + 1e-4 and x.min() >= -1e-4, "expect [0,1] range"
        x = self.normalization(x)  # [0,1] -> [-1,1], matching original dino_wm

        prefix_shape = x.shape[:-3]
        c, h, w = x.shape[-3:]

        # Collapse all leading dims into a single batch dimension for the base model
        prod_prefix = 1
        for d in prefix_shape:
            prod_prefix *= d
        x = x.reshape(prod_prefix, c, h, w)

        emb = self.base_model.forward_features(x)[self.feature_key]
        emb = emb.reshape(*prefix_shape, *emb.shape[1:])

        if self.postprocess == 'avg_pool':
            emb = torch.mean(emb, dim=-2)  # (...prefix, E)

        if self.latent_ndim == 1:
            emb = emb.unsqueeze(len(prefix_shape))

        return emb
