import numpy as np
from utils import min_max_scale


def test_min_max_scaling():
    x = [24, 11, 13.5]
    scaled_x = min_max_scale(x)
    assert np.max(scaled_x) == 1
    assert np.min(scaled_x) == 0

