import os

import json
import shutil
import cv2
import seaborn as sns
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

import SimpleITK as sitk
from src.datasets.seg_dataset_test import DatasetSeg
from src.aug.seg_aug import SameResAugsAlbu
from src.losses.seg_loss import SegFastLossCalculator
from src.metrics.seg_metrics import multiclass_dice_coefficient

from scipy import signal

import torch
from torch.utils.data import Dataset, DataLoader
import glob
import pydicom

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


DEST = "RSNA"
weights_path = f"{DEST}/backgrounds4/weights/best.pt"
model = YOLO(weights_path).to("cuda")


data = []
IMAGES_PATH = "data/train_images"

class DatasetYolo(Dataset):
    def __init__(
        self,
        path_to_images : str,
        ext : str = ".png"
    ):
        paths = glob.glob(f"{path_to_images}/*{ext}")
        paths_sorted = sorted(paths, key=lambda s: int(s.split("/")[-1].split(".")[0]))
        if len(paths_sorted) <= 400:
            koeff = 1
        else:
            koeff = int(len(paths_sorted) // 200)
        paths = []
        for i, p in enumerate(paths_sorted):
            if i % koeff == 0:
                paths.append(p)
                
        self.path_to_images = paths
       
    def __getitem__(self, i):
        image_path = self.path_to_images[i]
        image = cv2.imread(image_path).T
        return image
        
    def __len__(self):
        return len(self.path_to_images)
    
def TTA(model, batch : np.array, imgsz : int, conf : float):
    shape = batch.shape[0]
    confs, squares = np.zeros(shape), np.zeros(shape)
    for i in range(4):
        batch_rot = torch.rot90(batch, k=i, dims=[2, 3])
        
        preds = model.predict(batch_rot, imgsz=imgsz, conf=conf, verbose=False)
        for index, pred in enumerate(preds):
            if len(pred.boxes.conf) != 0:
                """
                беру только 1й предикт по конфиденсу и площади
                """
                pred_confs = pred.boxes.conf.detach().cpu().numpy()[0]
                confs[index] += pred_confs / 4
                pred_xywhn = pred.boxes.xywhn.detach().cpu().numpy()[:, 2:]
                squares[index] += (pred_xywhn[:, 0][0] * pred_xywhn[:, 1][0]) / 4
        
    return confs, squares

def getYoloScores(model, batch : np.array, imgsz : int = 512, conf : float = 0.01, is_tta : bool = True):
    if is_tta:
        confs, squares = TTA(model, batch, imgsz, conf)
        return confs, squares
    else:
        shape = batch.shape[0]
        confs, squares = np.zeros(shape), np.zeros(shape)
        preds = model.predict(batch, imgsz=imgsz, conf=conf, verbose=False)
        for index, pred in enumerate(preds):
            if len(pred.boxes.conf) != 0:
                """
                беру только 1й предикт по конфиденсу и площади
                """
                pred_confs = pred.boxes.conf.detach().cpu().numpy()[0]
                confs[index] += pred_confs
                pred_xywhn = pred.boxes.xywhn.detach().cpu().numpy()[:, 2:]
                squares[index] += pred_xywhn[:, 0][0] * pred_xywhn[:, 1][0]
        return confs, squares

for patient_id in tqdm(sorted(os.listdir(IMAGES_PATH))):
    patient_id = "10004"
    for paths in [glob.glob(os.path.join(IMAGES_PATH, patient_id, "*"))]:
        
        path = paths[1]#getBestScanId(paths)
        print(path)
        dataset = DatasetYolo(path_to_images = path, ext=".png")
        dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
        
        scores = []
        squares = []
        for batch in dataloader:
            score, square = getYoloScores(model, batch, conf=0.1, is_tta=False)
            scores.extend(score)
            squares.extend(square)
        sns.lineplot(scores)
        assert 1==0

        row = {"patient_id" : int(patient_id), "scores" : scores, "squares" : squares}
        data.append(row)