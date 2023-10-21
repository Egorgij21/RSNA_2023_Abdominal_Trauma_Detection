import os

import json
import shutil
import time
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ultralytics
import yaml
from PIL import Image
from tqdm.notebook import tqdm
from ultralytics import YOLO

BASE_FILENAME = "content/"

ultralytics.checks()


PATH_TO_DATA = "datasets/data.yaml"
FOLDS_ROOT = "loo_stained_wsi/"

PROJECT = "RSNA"
MODEL_V = "yolov8x.pt"

model = YOLO(MODEL_V)
model.train(
    task='detect',
    project=PROJECT,
    name=f"{MODEL_V}_backgrounds",
    # Random Seed parameters
    deterministic=True,
    seed=21,
    # Training parameters
    data=PATH_TO_DATA,
    single_cls=True,
    save=True,
    # save_period=10,
    pretrained=True,
    # pretrained=f"{PROJECT}/only_ds2_yolov8x-seg-fold{i}/weights/best.pt",
    imgsz=512,
    epochs=200,
    batch=45,
    workers=3,
    val=True,
    # fraction=0.8,
    device="cuda",
    dfl=1.5,  # 1.5 default, maybe 3 better
    box=7.5,  # 7.5 default
    retina_masks=True,

    # Optimization parameters
    lr0=1e-3,
    # patience=15,
    cos_lr=True,
    optimizer="AdamW",
    weight_decay=0.001,

    # Augmentation parameters
    overlap_mask=False,
    augment=True,
    mosaic=0.,
    mixup=0.,
    degrees=90.0,
    translate=0.3,
    scale=0.5,
    shear=15.0,
    perspective=0.0005,
    flipud=0.5,
    fliplr=0.5,
    hsv_h=0.04,  # image HSV-Hue augmentation (fraction) default 0.015
    hsv_s=0.4,  # (float) image HSV-Saturation augmentation (fraction) default 0.7
    hsv_v=0.3,  # (float) image HSV-Value augmentation (fraction) default 0.4
    # copy_paste=0.2, # strong augmentation
)
