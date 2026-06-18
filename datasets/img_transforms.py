from torchvision import transforms

def default_transform(img_size=224):
    # Images stay in [0, 1]; each encoder applies its own normalization
    # internally, so do not add a Normalize here.
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
        ]
    )