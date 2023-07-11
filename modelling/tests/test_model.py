import numpy as np
from model import FaceAgeDataset


def test_targets_scaled_zero_to_one():
    dataset = FaceAgeDataset(image_paths=["dummy/path.png", "dummy/path.png"],
                             ages=[34, 51],
                             life_expectancies=[50, 32])
    assert np.max(dataset.targets) == 1
    assert np.min(dataset.targets) == 0
