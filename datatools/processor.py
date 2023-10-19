import numpy as np
import pandas as pd
import torch
from typing import Tuple, List

import SimpleITK as sitk


def get_dicom_tags(path : str) -> pd.DataFrame:
    dicoms_tags = pd.read_parquet(path)
    dicoms_tags["z"]   = dicoms_tags.ImagePositionPatient.apply(lambda x: float(x.split(", ")[-1][:-1]))
    dicoms_tags["patient_id"] = dicoms_tags.path.apply(lambda x: x.split("/")[1])
    dicoms_tags["series_id"] = dicoms_tags.path.apply(lambda x: x.split("/")[2])
    dicoms_tags["slice_id"] = dicoms_tags.path.apply(lambda x: int(x.split("/")[-1].split(".")[0]))
    return dicoms_tags

def get_spacing(df : pd.DataFrame, patient_id : str, series_id : str) -> Tuple[float]:
    try:
        rows = df[(df.patient_id == patient_id) & (df.series_id == series_id)][["SliceThickness", "z", "slice_id"]][:2]
        thikness = rows.SliceThickness[:1].item()

        diff = rows.diff()[-1:]

        spacing = round((diff.z / ((diff.slice_id) + 1e-6)).item(), 3)
        return abs(spacing), thikness
    except:
        return 2.5, 2.5


def resize_sitk(image, output_size : Tuple[int], is_mask : bool = False):
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

def normalize_image(image : np.array) -> np.array:
    image_mean = image.mean()
    image_std = image.std()
    image = (image - image_mean) / image_std
    return image

def resize_cube_sitk(image : np.array, width : int, height : int, slice_size : int) -> np.array:
    image_sitk = sitk.GetImageFromArray(image)
    resized_image_sitk = resize_sitk(
            image_sitk,
            (width, height, slice_size),
            is_mask=False,
            )
    image = sitk.GetArrayFromImage(resized_image_sitk)
    return image

def prepare_cube_for_segmentation(image : np.array, width : int, height : int, slice_size : int) -> np.array:
    resized_image = resize_cube_sitk(image, width, height, slice_size)
    normalized_image = normalize_image(resized_image)
    image = np.expand_dims(normalized_image, 0)
    sample = {}
    sample["image"] = torch.from_numpy(image.astype(np.float32))
    return sample

def padding3D(image : sitk.Image, width, height, slice_size, full_zeroes=False) -> np.ndarray:
    shape = (width, height, slice_size)
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
        

def getOrganForClassification(image, width, height, slice_size):
    image_sitk = sitk.GetImageFromArray(image)
    image = padding3D(image_sitk, width, height, slice_size)
    image = np.stack([image, image, image])
    image = np.expand_dims(image, 0)
    sample = {}
    sample["image"] = torch.from_numpy(image.astype(np.float32))
    return sample