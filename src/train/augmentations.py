"""The augmentations used during training.
Feel free to experiment with any of the available augmentations:
https://albumentations.readthedocs.io/en/latest/index.html"""

import albumentations as albu


def det_train_augs(height: int, width: int) -> albu.Compose:
    return albu.Compose(
        [
            albu.RandomResizedCrop(512, 512),
            albu.ColorJitter(),
            albu.RandomGamma(),
            albu.Flip(),
            albu.Transpose(),
            albu.Rotate(),
            albu.Normalize(),
        ],
        bbox_params=albu.BboxParams(
            format="pascal_voc", label_fields=["category_ids"], min_visibility=0.2
        ),
    )


def det_val_augs(height: int, width: int) -> albu.Compose:
    return albu.Compose(
        [albu.Resize(height=height, width=width), albu.Normalize()],
        bbox_params=albu.BboxParams(
            format="pascal_voc", label_fields=["category_ids"], min_visibility=0.2
        ),
    )
