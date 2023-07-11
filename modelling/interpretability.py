import math
import numpy as np
import torch

from modelling.train import _generate_dataset, _get_train_test_split

DATASET_V2_MAX_TARGET = 0.918367326259613
DATASET_V2_MIN_TARGET = 0.0


def rescale_test_loss_to_years(mse, train_max_target, train_min_target):
    rmse = math.sqrt(mse)
    rescaled =  rmse * (train_max_target - train_min_target)
    return rescaled


def rescale_model_output_to_years_dataset_v2(output):
    return output


if __name__ == '__main__': 
    dataset = _generate_dataset()
    train_dataset, test_dataset = _get_train_test_split(dataset)

    train_targets = [row[3].item() for row in train_dataset]

    max_target = max(train_targets)
    min_target = min(train_targets)

    rescale = lambda x: rescale_model_output_to_years(x, max_target, min_target)

