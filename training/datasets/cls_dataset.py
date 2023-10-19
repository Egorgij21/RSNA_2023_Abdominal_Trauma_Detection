import glob
import os
from os.path import join

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def read_folder_sitk(folder, ext=".png"):
    paths = glob.glob(f"{folder}/*.png")
    paths = sorted(paths, key=lambda s: int(s.split("/")[-1].split(".")[0]))
    return sitk.ReadImage(paths)


def resize_sitk(image, output_size, is_mask=False):
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_spacing = [
        original_spacing[i] * (original_size[i] / output_size[i]) for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)

    resized_image = resampler.Execute(image)

    return resized_image


def padding3D(image: sitk.Image, shape, full_zeroes=False) -> np.ndarray:
    # If the original image is bigger than desired size, we resize both the image and the mask
    if any([image.GetSize()[i] > shape[i] for i in range(3)]):
        resize_factor = min([shape[i] / image.GetSize()[i] for i in range(3)])
        new_size = [int(image.GetSize()[i] * resize_factor) for i in range(3)]
        image = resize_sitk(image, new_size)

    def pad_helper(img_array: np.ndarray, shape: tuple, is_mask=False) -> np.ndarray:
        # Determine the value to pad with
        initial_value = np.min(img_array) if full_zeroes or is_mask else 0

        # Create an empty array with the desired shape
        pad_img_array = np.full(shape, initial_value, dtype=img_array.dtype)

        # Calculate the starting indices for the input image (to center it)
        offset = [(shape[i] - img_array.shape[i]) // 2 for i in range(3)]

        # Insert the input image into the empty array
        pad_img_array[
            offset[0] : offset[0] + img_array.shape[0],
            offset[1] : offset[1] + img_array.shape[1],
            offset[2] : offset[2] + img_array.shape[2],
        ] = img_array

        return pad_img_array

    padded_image = pad_helper(sitk.GetArrayFromImage(image).T, shape)

    return padded_image


class ClassifierDataset(Dataset):
    def __init__(
        self,
        target: str,
        mode: str,
        fold: int,
        path_to_images: str,
        path_to_meta_csv: str,
        shape: tuple,  # width, height, depth
    ):
        self.target = target
        meta = pd.read_csv(os.path.join(path_to_meta_csv))

        if mode == "train":
            self.meta = meta[meta.fold != fold]
        elif mode == 'eval':
            self.meta = meta[(meta.fold == fold)]
        else:
            self.meta = meta

        self.mode = mode
        # self.width, self.height, self.depth = shape
        self.shape = shape

        self.path_to_images = path_to_images

        self.cache = {}

        if self.is_train:
            assert len(self.meta[self.meta.fold == fold]) == 0
        elif mode == 'eval':
            assert len(self.meta[self.meta.fold != fold]) == 0

    @property
    def is_train(self):
        return self.mode == "train"

    def __len__(self):
        return len(self.meta)

    def flip_3d(self, image, axis):
        assert axis in [0, 1, 2], "axis should be in [0, 1, 2]"
        if axis == 0:
            return image[::-1]
        if axis == 1:
            return image[:, ::-1, :]
        return image[:, :, ::-1]

    def random_shift_3d(self, image, max_shift=(15, 30, 30)):
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

        return shifted_image

    def add_noise(self, image: np.ndarray, noise_factor: float) -> np.ndarray:
        sigma = noise_factor * (np.max(image) - np.min(image))
        gaussian_noise = np.random.normal(0, sigma, image.shape)
        noisy_img = image + gaussian_noise
        return noisy_img

    # def dropout(self, image: np.ndarray, p: float = 0.1) -> np.ndarray:
    #     return F.dropout(torch.Tensor(image), p=p).numpy()

    def augment_numpy(self, image, p=0.5):
        if np.random.rand() < p:
            image = self.flip_3d(image, 0)
        if np.random.rand() < p:
            image = self.flip_3d(image, 1)
        if np.random.rand() < p:
            image = self.flip_3d(image, 2)
        if np.random.rand() < p:
            image = self.random_shift_3d(image)
        if np.random.rand() < p:
            image = self.add_noise(image, 0.005)
        # if np.random.rand() < p:
        #     image = self.dropout(image, p=0.1)

        return image

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        patient_id = int(row.patient_id)

        image_path = join(self.path_to_images, str(patient_id))

        if self.target == "injury":  # binary target
            label = row.injury
        else:
            label = row[["healthy", "low", "high"]].values

        image_sitk = sitk.ReadImage(join(image_path, f"image.nii.gz"))  # normalized
        image = sitk.GetArrayFromImage(image_sitk)
        image = padding3D(image_sitk, shape=self.shape)

        if self.is_train:
            image = self.augment_numpy(image, 0.5)

        image = np.stack([image, image, image])

        #  w, h, depth -> 1, depth, w, h
        sample = {}

        sample["image"] = torch.from_numpy(image.astype(np.float32))
        sample["label"] = label if self.target == "injury" else torch.from_numpy(label)
        sample["patient_id"] = patient_id
        sample["label"] = label
        return sample
