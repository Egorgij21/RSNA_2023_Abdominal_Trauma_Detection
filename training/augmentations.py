from abc import ABC, abstractmethod

import albumentations as A
import cv2


class FractureAugmentations(ABC):
    @abstractmethod
    def get_train_augmentations(self) -> A.Compose:
        pass

    @abstractmethod
    def get_val_augmentations(self) -> A.Compose:
        pass


class SameResAugsAlbu(FractureAugmentations):
    def get_train_augmentations(self) -> A.Compose:
        return A.ReplayCompose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.5,
                ),
                A.HorizontalFlip(),
                A.OneOf(
                    [
                        A.GridDistortion(
                            border_mode=cv2.BORDER_CONSTANT, distort_limit=0.1
                        ),
                        A.ElasticTransform(border_mode=cv2.BORDER_CONSTANT),
                    ],
                    p=0.2,
                ),
            ]
        )

    def get_val_augmentations(self) -> A.Compose:
        return A.ReplayCompose([])
    

