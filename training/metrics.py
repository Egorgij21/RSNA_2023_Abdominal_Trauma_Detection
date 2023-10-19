import numpy as np
import pandas as pd
import pandas.api.types
import torch
import sklearn
from sklearn.metrics import log_loss

class ParticipantVisibleError(Exception):
    pass

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
    





def multiclass_dice_coefficient(output, target, num_classes, eps=1e-8):
    """
    Calculates the Dice coefficient for multiclass segmentation.

    Args:
        output (torch.Tensor): The predicted segmentation of shape (N, C, Z, X, Y).
        target (torch.Tensor): The ground truth segmentation of shape (N, Z, X, Y).
        num_classes (int): The number of classes.
        eps (float): A small number to avoid division by zero.

    Returns:
        dice_coeff (float): The Dice coefficient for multiclass segmentation.
    """
    dice_scores = []

    for i in range(num_classes):
        pred_class = (output.argmax(dim=1) == i).float()
        true_class = (target == i).float()
        intersection = (pred_class * true_class).sum(dim=(1, 2, 3))
        union = pred_class.sum(dim=(1, 2, 3)) + true_class.sum(dim=(1, 2, 3))
        dice_score = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice_score)

    dice_coeff = torch.stack(dice_scores, dim=0).mean()

    return dice_coeff