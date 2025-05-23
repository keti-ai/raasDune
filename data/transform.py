import torch
from torchvision.transforms import v2 as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NORMALIZE = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def get_train_transform(image_size=224, rrc_scale=(0.08, 1.0), color_aug=True):
    transforms = [
        T.ToImage(),
        T.RandomResizedCrop(
            image_size,
            scale=rrc_scale,
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.RandomHorizontalFlip(p=0.5),
    ]

    if color_aug:
        transforms.extend(
            [
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.2),
                T.ToDtype(torch.float32, scale=True),
                T.RandomApply([T.GaussianBlur(kernel_size=9, sigma=(0.1, 5.0))], p=0.2),
                T.RandomSolarize(threshold=0.5, p=0.2),
            ]
        )
    else:
        transforms.extend(
            [
                T.ToDtype(torch.float32, scale=True),
            ]
        )

    transforms.append(NORMALIZE)
    return T.Compose(transforms)


def get_test_transform(image_size, normalize=NORMALIZE, center_crop_size=None):
    if center_crop_size is None:
        center_crop_size = image_size

    transforms = [
        T.ToImage(),
        T.Resize(
            image_size,
            interpolation=T.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        T.CenterCrop(center_crop_size),
        T.ToDtype(torch.float32, scale=True),
        normalize,
    ]

    return T.Compose(transforms)
