import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class resnet18(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        unit_norm: bool = False,
        output_dim: int = 512,  # fixed for resnet18; accepted for config compatibility
        n_patches: int = 1,     # fixed for resnet18 (global pool); for config compatibility
    ):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unit_norm = unit_norm

        self.latent_ndim = 1
        self.emb_dim = 512
        self.output_dim = 512
        self.n_patches = 1
        self.patch_size = None
        self.name = "resnet"

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.normalize(x)
        out = self.resnet(x)
        out = self.flatten(out)
        if self.unit_norm:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        out = out.unsqueeze(1)
        return out


class Resnet18Patches(nn.Module):
    """ResNet18 patch encoder.

    Returns patch tokens ``(..., n_patches, output_dim)`` from an intermediate
    ResNet18 layer, with the encoder-contract attributes (``latent_ndim == 2``,
    ``emb_dim``, ``name``) so it works in both the world-model encoder slot and
    the BC policy.

    Expects pixels in ``[0, 1]`` and
    applies ImageNet normalization internally before the backbone.
    """

    def __init__(
        self,
        pretrained: bool = True,
        output_dim: int = 256,
        unit_norm: bool = False,
        ckpt_path: Optional[str] = None,
        return_layers: Tuple[str, ...] = ("layer3",),
        n_patches: Optional[int] = None,
    ):
        super().__init__()
        base_model = torchvision.models.resnet18(pretrained=pretrained)

        # ResNet18 nodes: conv1, bn1, relu, maxpool, layer1..layer4
        self.return_layers = list(return_layers)
        from torchvision.models.feature_extraction import create_feature_extractor
        self.backbone = create_feature_extractor(base_model, return_nodes=self.return_layers)

        self.pretrained = pretrained
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.unit_norm = unit_norm
        self.output_dim = output_dim

        # dino_wm encoder contract
        self.emb_dim = output_dim
        self.latent_ndim = 2
        self.name = "resnet_patch"  # must NOT contain "dino" (see VWorldModel)
        self.patch_size = None

        # Determine patch count + per-layer 1x1 projections via a dummy pass
        self.n_patches = 0
        self.projections = nn.ModuleDict()
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        for layer_name in self.return_layers:
            feat = features[layer_name]
            c, h, w = feat.shape[1], feat.shape[2], feat.shape[3]
            self.n_patches += h * w
            if c != self.output_dim:
                self.projections[layer_name] = nn.Conv2d(c, self.output_dim, kernel_size=1)
            else:
                self.projections[layer_name] = nn.Identity()

        if n_patches is not None:
            assert n_patches == self.n_patches, (
                f"configured n_patches={n_patches} but layers produce {self.n_patches}"
            )

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state_dict = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
            self.load_from_resnet18_state_dict(state_dict)

    def load_from_resnet18_state_dict(self, state_dict, strict: bool = False):
        """Map a legacy global-pooling resnet18 encoder checkpoint (Sequential
        layout, e.g. a pretrained ``encoder.pt``) onto this named backbone."""
        idx_map = {
            "0": "conv1", "1": "bn1",  # 2 (relu), 3 (maxpool) are stateless
            "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4",
        }
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("module.resnet."):
                k = k[len("module."):]
            if not k.startswith("resnet."):
                continue
            parts = k.split(".")
            seq_idx = parts[1]
            if seq_idx not in idx_map:
                continue
            new_key = ".".join(["backbone", idx_map[seq_idx]] + parts[2:])
            new_state[new_key] = v
        return self.load_state_dict(new_state, strict=strict)

    def forward(self, x):
        dims = len(x.shape)
        squeeze_batch = dims == 3
        if squeeze_batch:
            x = x.unsqueeze(0)

        leading_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        x = self.normalize(x)  # expects pixels in [0, 1]

        features = self.backbone(x)
        token_list = []
        for layer_name in self.return_layers:
            feat = self.projections[layer_name](features[layer_name])  # (Bf, D, H, W)
            tokens = feat.flatten(2).transpose(1, 2)  # (Bf, H*W, D)
            token_list.append(tokens)
        out = torch.cat(token_list, dim=1)  # (Bf, total_patches, D)

        if self.unit_norm:
            out = F.normalize(out, p=2, dim=-1)

        out = out.reshape(*leading_shape, out.shape[1], out.shape[2])
        if squeeze_batch:
            out = out.squeeze(0)
        return out


class resblock(nn.Module):
    # this implementation assumes square images
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=32):
        super(resblock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample

        padding = int((kernel_size - 1) / 2)

        if resample == "down":
            self.skip = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
            )
            self.conv1 = nn.Conv2d(
                input_dim, input_dim, kernel_size, padding=padding, bias=False
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, kernel_size, padding=padding),
                nn.MaxPool2d(2, stride=2),
            )
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample is None:
            self.skip = nn.Conv2d(input_dim, output_dim, 1)
            self.conv1 = nn.Conv2d(
                input_dim, output_dim, kernel_size, padding=padding, bias=False
            )
            self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size, padding=padding)
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)

        self.leakyrelu1 = nn.LeakyReLU()
        self.leakyrelu2 = nn.LeakyReLU()

    def forward(self, x):
        if (self.input_dim == self.output_dim) and self.resample is None:
            idnty = x
        else:
            idnty = self.skip(x)

        residual = x
        residual = self.conv1(residual)
        residual = self.bn1(residual)
        residual = self.leakyrelu1(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.leakyrelu2(residual)

        return idnty + residual


class SmallResNet(nn.Module):
    def __init__(self, output_dim=512):
        super(SmallResNet, self).__init__()

        self.hw = 224

        # 3x224x224
        self.rb1 = resblock(3, 16, 3, resample="down", hw=self.hw)
        # 16x112x112
        self.rb2 = resblock(16, 32, 3, resample="down", hw=self.hw // 2)
        # 32x56x56
        self.rb3 = resblock(32, 64, 3, resample="down", hw=self.hw // 4)
        # 64x28x28
        self.rb4 = resblock(64, 128, 3, resample="down", hw=self.hw // 8)
        # 128x14x14
        self.rb5 = resblock(128, 512, 3, resample="down", hw=self.hw // 16)
        # 512x7x7
        self.maxpool = nn.MaxPool2d(7)
        # 512x1x1
        self.flat = nn.Flatten()

    def forward(self, x):
        dims = len(x.shape)
        orig_shape = x.shape
        if dims == 3:
            x = x.unsqueeze(0)
        elif dims > 4:
            # flatten all dimensions to batch, then reshape back at the end
            x = x.reshape(-1, *orig_shape[-3:])
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.rb5(x)
        x = self.maxpool(x)
        out = x.flatten(start_dim=-3)
        if dims == 3:
            out = out.squeeze(0)
        elif dims > 4:
            out = out.reshape(*orig_shape[:-3], -1)
        return out
