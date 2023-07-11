import numpy as np


def min_max_scale(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def undo_min_max_scaling(x, min_val, max_val):
    return x * (max_val - min_val) + min_val
