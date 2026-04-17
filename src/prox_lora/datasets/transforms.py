import PIL.Image
import torch
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import v2


def _to_rgb(img: PIL.Image.Image) -> PIL.Image.Image:
    """Convert image to RGB."""
    return img.convert("RGB") if img.mode != "RGB" else img


def default_transform(target_image_size: int = 224) -> v2.Transform:
    """
    Default transform for image classification (as typically taken for ImageNet).

    Essentially the same as
        `torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms(antialias=True)`
    or
        `timm.data.create_transform(crop_pct=1)`.
    """
    return v2.Compose(
        [
            v2.Lambda(_to_rgb),
            v2.ToImage(),
            v2.Resize([target_image_size], antialias=True),  # Resize so that min dimension is image_size.
            v2.CenterCrop(target_image_size),  # Crop rectangle centrally to square.
            v2.ToDtype(torch.float32, scale=True),  # Convert to float and scale to 0..1.
            v2.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def default_clip_transform(target_image_size: int = 224) -> v2.Transform:
    """
    Default transform for CLIP models.

    Essentially the same as
    create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")[1]
        `create_model_from_pretrained("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")[1]`
    or
        `open_clip.transform.image_transform_v2(PreprocessCfg(), is_train=False)`
    or
        `default_transform()` but with different normalization and with bicubic interpolation.
    """
    return v2.Compose(
        [
            v2.Lambda(_to_rgb),
            v2.ToImage(),
            v2.Resize(
                [target_image_size], antialias=True, interpolation=v2.InterpolationMode.BICUBIC
            ),  # Resize so that min dimension is image_size.
            v2.CenterCrop(target_image_size),  # Crop rectangle centrally to square.
            v2.ToDtype(torch.float32, scale=True),  # Convert to float and scale to 0..1.
            v2.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
        ]
    )
    # For training they use
    #   RandomResizedCrop(starget_image_size, scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic, antialias=True)
