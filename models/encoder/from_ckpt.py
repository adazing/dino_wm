import torch


def from_ckpt(f: str, output_dim: int, n_patches: int):
    """Load a whole pretrained encoder object saved with torch.save(model).

    output_dim / n_patches are accepted so they can be referenced as config
    metadata (e.g. by the policy/decoder) without being applied here.
    """
    model = torch.load(f, weights_only=False)
    # Expose dino_wm encoder-contract attributes if the loaded object lacks them.
    if not hasattr(model, "emb_dim"):
        model.emb_dim = output_dim
    if not hasattr(model, "latent_ndim"):
        model.latent_ndim = 2
    if not hasattr(model, "name"):
        model.name = "from_ckpt"
    if not hasattr(model, "patch_size"):
        model.patch_size = None
    return model
