from torch.utils.data import WeightedRandomSampler
import numpy as np


targets = train_dataset.meta[["healthy", "low", "high"]].values.argmax(1)
class_counts = np.bincount(targets)
class_weights = 1. / class_counts
weights = class_weights[targets]
sampler = WeightedRandomSampler(weights, len(weights))