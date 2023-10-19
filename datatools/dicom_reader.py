import os

import pydicom

import numpy as np

def get_files(folder, ext=''):
    paths = []
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isfile(item_path) and item.endswith(ext):
            paths.append(item_path)
        elif os.path.isdir(item_path):
            paths.extend(get_files(item_path, ext))
    return paths

def standardize_pixel_array(dcm: pydicom.dataset.FileDataset) -> np.ndarray:
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        pixel_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
    return pixel_array

def read_dicom(path: str, fix_monochrome: bool = True) -> np.ndarray:
    dicom = pydicom.dcmread(path)
    data = standardize_pixel_array(dicom)
    
    # find rescale params
    if ("RescaleIntercept" in dicom) and ("RescaleSlope" in dicom):
        intercept = float(dicom.RescaleIntercept)
        slope = float(dicom.RescaleSlope)

    # find clipping params
    center = int(dicom.WindowCenter)
    width = int(dicom.WindowWidth)
    
    low = center - width / 2
    high = center + width / 2

    data = (data * slope) + intercept
    
    data = np.clip(data, low, high)

    data = data - np.min(data)
    data = data / (np.max(data) + 1e-5)
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1.0 - data
    return (data * 255).astype(np.uint8)

def get_cube(path : str, koeff : int) -> np.ndarray:
    dicom_paths = get_files(path, ext='.dcm')
    sorted_dicom_paths = sorted(dicom_paths, key=lambda s: int(s.split("/")[-1].split(".")[0]))
    cube = []
    for i, image_path in enumerate(sorted_dicom_paths):
        if i % koeff != 0:
            continue
        image = read_dicom(image_path)
        cube.append(image)
    return np.stack(cube)