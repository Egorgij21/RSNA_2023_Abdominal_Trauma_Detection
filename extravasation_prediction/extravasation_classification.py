import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
import sklearn

from imblearn.ensemble import BalancedRandomForestClassifier

N_SPLITS = 4
skf = StratifiedKFold(n_splits=N_SPLITS, random_state=21, shuffle=True)


path = "path/to/yolo_prediction/dataset.csv"
df = pd.read_csv(path)

feature_cols = ['scores_mean',
       'scores_max', 'scores_median', 'scores_std',
       'scores_std_not_null', 'shape', "feature_1", 
       'squares_max', 'squares_median',
       'squares_std_not_null',]

class Metric():
    def __init__(
        self,
        label : int
    ):
        self.label = label
        self.label_to_class = {1 : 'liver',
                               2 : 'spleen',
                               3 : 'kidney',
                               4 : 'bowel',
                               5 : 'extravasation'}
        
        self.label_to_weights = {1 : [1, 2, 4],
                                 2 : [1, 2, 4],
                                 3 : [1, 2, 4],
                                 4 : [1, 2],
                                 5 : [1, 6]}
        
        self.label_weights = self.label_to_weights[label]
        
        
    def get_weights(self, true):
        if len(self.label_weights) == 2:
            weights = [self.label_weights[0] if i == 0 else self.label_weights[1] for i in true]
        else:
            weights = []
            for row in true:
                i = np.where(row == 1)[0][0]
                weights.append(self.label_weights[i])
        return weights
    
    

import pandas.api.types
import sklearn
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticipantVisibleError(Exception):
    pass


    def normalize_probabilities_to_one(self, pred: np.array) -> np.array:
        # Normalize the sum of each row's probabilities to 100%.
        # 0.75, 0.75 => 0.5, 0.5
        # 0.1, 0.1 => 0.5, 0.5
        row_totals = pred.sum(axis=1)
        pred_new = np.moveaxis(pred, 0, -1)
        if row_totals.min() == 0:
            raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
        for i in range(pred.shape[1]):
            pred_new[i] /= row_totals
        pred_new = np.moveaxis(pred_new, 0, -1)
        return pred_new
    
    
    def get_score(self, true, pred, normalize=True):
        weights = self.get_weights(true)
        pred = self.normalize_probabilities_to_one(pred)
        score = log_loss(
                        y_true=true,
                        y_pred=pred,
                        normalize=normalize,
                        sample_weight=weights
                        )
        return score
    
    def get_class(self):
        return self.label_to_class[self.label]
    


Score = Metric(label = 5)
val_scores = []

for i in range(N_SPLITS):
    X_train = df[feature_cols][df.fold != i]
    y_train = df[df.fold != i].label
    X_val = df[feature_cols][df.fold == i]
    y_val = df[df.fold == i].label
    
    LR = LogisticRegression(random_state=21,
                            penalty='elasticnet',
                            class_weight={0 : 1, 1 : 6},
                             solver="saga",
                            l1_ratio=0.9)

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
                                         class_weight={0 : 1, 1 : 5},
                                         ccp_alpha=0.,
                                         max_samples=None
                                        )

    fit_LR = LR.fit(X_train, y_train)
    fit_BRF = BRF.fit(X_train, y_train)
    
    pred = np.array(0.5 * fit_BRF.predict(X_val) + 0.5 * fit_LR.predict(X_val), dtype = np.uint8)
    f1 = f1_score(y_val, pred)
    
    true = np.array(y_val)
    pred_LR = np.array(fit_LR.predict_proba(X_val))
    pred_BRF = np.array(fit_BRF.predict_proba(X_val))
    pred = 0.5 * pred_BRF + 0.5 * pred_LR
    val_score = Score.get_score(true, pred)
    val_scores.append(val_score)
    
    print(f'fold: {i}')
    print(f'f1 val score: {f1}')
    print(f'w_logloss val score: {val_score}')
    print()
    
print(f"mean val score: {np.mean(val_scores)} +- {2 * np.std(val_scores)}")