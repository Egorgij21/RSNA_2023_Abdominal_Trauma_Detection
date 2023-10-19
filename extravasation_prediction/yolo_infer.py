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

import torch
from torch.utils.data import Dataset, DataLoader

import glob
import joblib

from imblearn import BalancedRandomForestClassifier


class DatasetYoloKaggle(Dataset):
    def __init__(
        self,
        cube : np.array,
    ):
        self.cube = cube
       
    def __getitem__(self, i):
        image = self.cube[i]
        if image.shape[0] != 512 or image.shape[1] != 512:
            image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).T
        return image
        
    def __len__(self):
        return self.cube.shape[0]
    
class ExtravasationPrediction():
    def __init__(
        self,
        path_to_yolo_weights : str,
        path_to_pred_models : str,
        device : str
    ):
        self.YOLO = YOLO(path_to_yolo_weights).to(device)
        self.prediction_models = self.initializeExtravasationModels(path_to_pred_models)

    def initializeExtravasationModels(self, path_to_models : str):
        lineag_regressions = []
        shifted_lineag_regressions = []
        random_forests = []
        shifted_random_forests = []
        for model_path in glob.glob(os.path.join(path_to_models, "*.pkl")):
            if model_path.split("/")[-1].split("_")[0] == "LR":
                if model_path.split("/")[-1].split("_")[1] != "shifted":
                    LR = joblib.load(model_path)
                    lineag_regressions.append(LR)
                else:
                    LR = joblib.load(model_path)
                    shifted_lineag_regressions.append(LR)
                
        full_train_data = pd.read_csv("/kaggle/input/extr-models/data/data.csv")
        shifted_train_data = pd.read_csv("/kaggle/input/extr-models/data/shifted_labels.csv")
        feature_cols = ['scores_mean', 'scores_max', 'scores_median', 'scores_std', 'scores_std_not_null',
                        'shape', "feature_1", 'squares_max', 'squares_median', 'squares_std_not_null']
        BRF = BalancedRandomForestClassifier(n_estimators=100,
                                     criterion="gini",
                                     max_depth=None,
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.,
                                     max_features='sqrt',
                                     max_leaf_nodes=None,
                                     min_impurity_decrease=0.,
                                     bootstrap=True,
                                     oob_score=False,
                                     sampling_strategy="auto",
                                     replacement=False,
                                     random_state=21,
                                     verbose=0,
                                     warm_start=False,
                                     class_weight={0 : 1, 1 : 6},
                                     ccp_alpha=0.,
                                     max_samples=None
                                    )
        N_SPLITS = len(full_train_data.fold.unique())
        for i in range(N_SPLITS):
            X_train = full_train_data[full_train_data.fold != i][feature_cols]
            y_train = full_train_data[full_train_data.fold != i]["label"]
            X_val = full_train_data[full_train_data.fold == i][feature_cols]
            y_val = full_train_data[full_train_data.fold == i]["label"]
            fit_BRF = BRF.fit(X_train, y_train)
            random_forests.append(fit_BRF)
        fit_BRF = BRF.fit(full_train_data[feature_cols], full_train_data["label"])
        random_forests.append(fit_BRF)
            
        N_SPLITS = len(shifted_train_data.fold.unique())
        for i in range(N_SPLITS):
            X_train = shifted_train_data[shifted_train_data.fold != i][feature_cols]
            y_train = shifted_train_data[shifted_train_data.fold != i]["label"]
            X_val = shifted_train_data[shifted_train_data.fold == i][feature_cols]
            y_val = shifted_train_data[shifted_train_data.fold == i]["label"]
            fit_BRF = BRF.fit(X_train, y_train)
            shifted_random_forests.append(fit_BRF)
        fit_BRF = BRF.fit(shifted_train_data[feature_cols], shifted_train_data["label"])
        shifted_random_forests.append(fit_BRF)

        return (lineag_regressions, random_forests, shifted_lineag_regressions, shifted_random_forests)
    
    def TTA(self, model, batch : np.array, imgsz : int, conf : float):
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
    
    def getYoloScores(self, model, batch : np.array, imgsz : int = 512, conf : float = 0.01, is_tta : bool = False):
        if is_tta:
            confs, squares = self.TTA(model, batch, imgsz, conf)
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

    def getYoloPredictions(self, cube : np.array, batch_size : int, is_tta : bool = False):
        model = self.YOLO
        dataset = DatasetYoloKaggle(cube)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        scores = []
        squares = []
        for batch in dataloader:
            score, square = self.getYoloScores(model, batch, conf=0.1, is_tta=is_tta)
            scores.extend(score)
            squares.extend(square)
        return scores, squares
    
    def f(self, x, th = 0.35):
        counter = 0
        m = 0
        for i in x:
            if i > th:
                counter += 1
            else:
                counter = 0
            if counter > m:
                m = counter
        return m / len(x)

    def getFeatures(self, scores : np.array, squares : np.array):
        scores = np.array(scores)
        squares = np.array(squares)
        scores_not_null = scores[scores!=0.0]
        scores_mean = scores_not_null.mean()
        scores_max = scores.max()
        scores_median = np.median(scores_not_null)
        scores_std = scores.std()
        scores_std_not_null = scores_not_null.std()
        shape = len(scores) / 500
        my_feature = (self.f(scores, 0.35) * scores_max / (scores_not_null.min() + 1e-6)) * (scores_std_not_null/(scores_std + 1e-6))
        squares_max = squares.max()
        squares_median = np.median(squares[squares!=0.0])
        squares_std_not_null = squares[squares!=0.0].std()
        return pd.DataFrame({ 'scores_mean' : scores_mean, 'scores_max' : scores_max, 'scores_median' : scores_median,
                 'scores_std' : scores_std, 'scores_std_not_null' : scores_std_not_null, 'shape' : shape,
                 'feature_1' : my_feature, 'squares_max' : squares_max, 'squares_median' : squares_median, 'squares_std_not_null' : squares_std_not_null}, index=[0])
        

    def getExtravasationPrediction(self, cube : np.array, batch_size : int, use_shifted_data : bool = False):
        scores, squares = self.getYoloPredictions(cube, batch_size, is_tta=False)
        features = self.getFeatures(scores, squares)
        
        NUM_MODELS = 5
        prediction_models = self.prediction_models
        linear_regressions = prediction_models[0]
        random_forests = prediction_models[1]
        shifted_linear_regressions = prediction_models[2]
        shifted_random_forests = prediction_models[3]
        LR_scores = np.array([0., 0.])
        RF_scores = np.array([0., 0.])
        for fold_num in range(NUM_MODELS):
            LR = linear_regressions[fold_num]
            RF = random_forests[fold_num]
            LR_pred = LR.predict_proba(features)[0]
            RF_pred = RF.predict_proba(features)[0]
            LR_scores += LR_pred / NUM_MODELS
            RF_scores += RF_pred / NUM_MODELS
        score = 0.5 * LR_scores + 0.5 * RF_scores
        if use_shifted_data:
            LR_scores_shifted = np.array([0., 0.])
            RF_scores_shifted = np.array([0., 0.])
            for fold_num in range(NUM_MODELS):
                LR = shifted_linear_regressions[fold_num]
                RF = shifted_random_forests[fold_num]
                LR_pred = LR.predict_proba(features)[0]
                RF_pred = RF.predict_proba(features)[0]
                LR_scores_shifted += LR_pred / NUM_MODELS
                RF_scores_shifted += RF_pred / NUM_MODELS
            score_shifted = 0.5 * LR_scores_shifted + 0.5 * RF_scores_shifted
            score = 0.8 * score + 0.2 * score_shifted
        return score
    


EP = ExtravasationPrediction(path_to_yolo_weights = "extr_models/best_yolo.pt",
                             path_to_pred_models = "extr_models",
                             device = "cuda")