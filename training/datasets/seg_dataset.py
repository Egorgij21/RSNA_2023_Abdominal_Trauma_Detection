import glob
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class DatasetSeg(Dataset):
    def __init__(
        self,
        mode: str,
        fold: int,
        transforms: A.Compose,
        slice_size: int,
        path_to_images: str,
        path_to_meta_csv: str,
        path_to_masks: str,
        shape=(256, 256),
    ):
        meta = pd.read_csv(os.path.join(path_to_meta_csv))

        if mode == "train":
            self.meta = meta[meta.fold != fold]
        else:
            self.meta = meta[(meta.fold == fold)]

        self.mode = mode
        self.transforms = transforms
        self.slice_size = slice_size
        if self.slice_size % 16 != 0:
            self.slice_size = self.slice_size + self.slice_size % 16
            print(f"new slice size: {self.slice_size}")
        self.path_to_images = path_to_images
        self.path_to_masks = path_to_masks
        self.height = shape[0]
        self.width = shape[1]

        self.resampler = sitk.ResampleImageFilter()
        self.resampler.SetSize((self.slice_size, self.height, self.width))

        self.cache = {}

        if self.is_train:
            assert len(self.meta[self.meta.fold == fold]) == 0
        else:
            assert len(self.meta[self.meta.fold != fold]) == 0

    def flip_3d(self, image, mask, axis):
        assert axis in [0, 1, 2], "axis should be in [0, 1, 2]"
        if axis == 0:
            return image[::-1], mask[::-1]
        if axis == 1:
            return image[:, ::-1, :], mask[:, ::-1, :]
        return image[:, :, ::-1], mask[:, :, ::-1]

    # def random_shift_cycle_3d(self, image, mask, max_shift=(10, 20, 20)):
    #     dx = np.random.randint(-max_shift[0], max_shift[0])
    #     dy = np.random.randint(-max_shift[1], max_shift[1])
    #     dz = np.random.randint(-max_sаhift[2], max_shift[2])

    #     shifted_image = np.roll(image, shift=dx, axis=0)
    #     shifted_image = np.roll(shifted_image, shift=dy, axis=1)
    #     shifted_image = np.roll(shifted_image, shift=dz, axis=2)

    #     shifted_mask = np.roll(mask, shift=dx, axis=0)
    #     shifted_mask = np.roll(shifted_mask, shift=dy, axis=1)
    #     shifted_mask = np.roll(shifted_mask, shift=dz, axis=2)

    #     return shifted_image, shifted_mask

    def random_shift_3d(self, image, mask, max_shift=(15, 30, 30)):
        dx = np.random.randint(-max_shift[0], max_shift[0])
        dy = np.random.randint(-max_shift[1], max_shift[1])
        dz = np.random.randint(-max_shift[2], max_shift[2])

        def shift_array(data, dx, dy, dz):
            shifted_data = np.zeros_like(data)

            x_slices_src = (max(0, -dx), min(data.shape[0], data.shape[0] - dx))
            y_slices_src = (max(0, -dy), min(data.shape[1], data.shape[1] - dy))
            z_slices_src = (max(0, -dz), min(data.shape[2], data.shape[2] - dz))

            x_slices_dest = (max(0, dx), min(data.shape[0], data.shape[0] + dx))
            y_slices_dest = (max(0, dy), min(data.shape[1], data.shape[1] + dy))
            z_slices_dest = (max(0, dz), min(data.shape[2], data.shape[2] + dz))

            shifted_data[
                x_slices_dest[0] : x_slices_dest[1],
                y_slices_dest[0] : y_slices_dest[1],
                z_slices_dest[0] : z_slices_dest[1],
            ] = data[
                x_slices_src[0] : x_slices_src[1],
                y_slices_src[0] : y_slices_src[1],
                z_slices_src[0] : z_slices_src[1],
            ]

            return shifted_data

        shifted_image = shift_array(image, dx, dy, dz)
        shifted_mask = shift_array(mask, dx, dy, dz)

        return shifted_image, shifted_mask

    def augment_numpy(self, image, mask, p=0.5):
        if np.random.rand() < p:
            image, mask = self.flip_3d(image, mask, 0)
        if np.random.rand() < p:
            image, mask = self.flip_3d(image, mask, 1)
        if np.random.rand() < p:
            image, mask = self.flip_3d(image, mask, 2)

        if np.random.rand() < p:
            image, mask = self.random_shift_3d(image, mask)

        return image, mask

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        patient_id = int(row.patient_id)
        patient_scan_id = int(row.series_id)
        image_path = os.path.join(
            self.path_to_images, str(patient_id), str(patient_scan_id)
        )
        mask_path = os.path.join(self.path_to_masks, str(patient_scan_id))

        image_sitk = self.read_sitk(image_path)  # sitk shape: w, h, depth
        mask_sitk = self.read_sitk(mask_path)

        resized_image_sitk = self.resize_sitk(
            image_sitk,
            (self.width, self.height, self.slice_size),
            is_mask=False,
        )
        resized_mask_sitk = self.resize_sitk(
            mask_sitk,
            (self.width, self.height, self.slice_size),
            is_mask=True,
        )

        # this func change shape to depth, w, h
        image = sitk.GetArrayFromImage(resized_image_sitk)
        mask = sitk.GetArrayFromImage(resized_mask_sitk)

        if patient_scan_id not in self.cache:
            image_mean = image.mean()
            image_std = image.std()
            image_max = image.max()
            sums = [0]
            for label in range(1, 6):
                sums.append((mask == label).sum())
            self.cache[patient_scan_id] = (image_mean, image_std, image_max, sums)
        else:
            image_mean, image_std, image_max, sums = self.cache[patient_scan_id]

        image = (image - image_mean) / image_std

        if self.is_train:
            image, mask = self.augment_numpy(image, mask, 0.5)

        image = np.expand_dims(image, 0)

        #  w, h, depth -> 1, depth, w, h
        sample = {}

        sample["image"] = torch.from_numpy(image.astype(np.float32))
        sample["mask"] = torch.from_numpy(mask.astype(np.uint8))
        sample["patient_id"] = patient_id
        sample["scan_id"] = patient_scan_id

        return sample

    def read_sitk(self, folder, ext=".png"):
        paths = glob.glob(f"{folder}/*.png")
        paths = sorted(paths, key=lambda s: int(s.split("/")[-1].split(".")[0]))
        return sitk.ReadImage(paths)

    def resize_sitk(self, image, output_size, is_mask=False):
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        new_spacing = [
            original_spacing[i] * (original_size[i] / output_size[i]) for i in range(3)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(output_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(
            sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
        )

        resized_image = resampler.Execute(image)

        return resized_image
    


    @property
    def is_train(self):
        return self.mode == "train"

    def __len__(self):
        return len(self.meta)
