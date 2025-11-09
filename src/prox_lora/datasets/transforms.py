import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import v2


def default_transform(target_image_size: int = 224) -> v2.Transform:
    """Default transform for image classification (as typically taken for ImageNet).

    Essentially the same as torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True).
    or timm.data.create_transform(crop_pct=1).
    """
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize([target_image_size], antialias=True),  # Resize so that min dimension is image_size.
            v2.CenterCrop(target_image_size),  # Crop rectangle centrally to square.
            v2.ToDtype(torch.float32, scale=True),  # Convert to float and scale to 0..1.
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
