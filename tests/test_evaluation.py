import numpy as np
import torch
from torch.utils.data import DataLoader

from life_expectancy.modelling.utils import min_max_scale, set_seed
from life_expectancy.modelling.config import CONFIG
from life_expectancy.modelling.data import (get_val_test_datasets,
                                            get_train_dataset)
from life_expectancy.analysis.eval import (get_mean_guess_stats,
                                           get_model_stats)
from life_expectancy.modelling.model import ResNet


CONFIG["BATCH_SIZE"] = 1
expected_train_targets = np.array([0.13235294, 0.22058824, 1., 0.,
                                   0.01470588,  0.79411765])
expected_val_targets = np.array([0., 1., 0.27027027, 0.16216216,
                                 0.45945946, 0.27027027])


class MeanPredictingModel(ResNet):
    def forward(self, img, age):
        mean = np.mean(expected_train_targets)
        return torch.Tensor([mean]).unsqueeze(-1)


def test_mean_guess_stats():
    dummy_train_ds = get_dummy_train_ds()
    dummy_val_ds = get_dummy_val_ds()
    dummy_val_dl = get_dummy_val_dl()
    dataset_dict = {"datasets": {"train": dummy_train_ds,
                                 "validation": dummy_val_ds},
                    "dataloaders": {"validation": dummy_val_dl}}

    expected_mae = get_expected_mae(dummy_train_ds,
                                    dummy_val_ds)
    mae = get_mean_guess_stats(dataset_dict)["mae"]
    assert np.isclose(mae, expected_mae, atol=1e-5)

    model = MeanPredictingModel()
    model_mae = get_model_stats(model, dataset_dict)["mae"]
    assert np.isclose(model_mae, expected_mae, atol=1e-5)


def get_expected_mae(dummy_train_ds, dummy_val_ds):
    train_targets = dummy_train_ds.targets
    val_targets = dummy_val_ds.targets
    mean_target = 0.3602941183333333
    expected_mae = 0.24622416557057059 # mean abs. error on the val
    scaling_factor = 37 #max - min of the DELTAS
    expected_mae = expected_mae * scaling_factor
    return expected_mae


def get_dummy_val_dl():
    dummy_val_ds = get_dummy_val_ds()
    dl = DataLoader(dummy_val_ds, batch_size=1, shuffle=False)
    return dl


def get_dummy_train_dl():
    dummy_val_ds = get_dummy_train_ds()
    dl = DataLoader(dummy_val_ds, batch_size=1, shuffle=False)
    return dl


def get_dummy_val_ds():
    CONFIG["DATA_FRACTION"] = 1/500
    dummy_val_ds, _ = get_val_test_datasets(CONFIG)
    assert all(np.isclose(dummy_val_ds.targets,
                      expected_val_targets,
                      atol=0.00001))
    return dummy_val_ds


def get_dummy_train_ds():
    CONFIG["DATA_FRACTION"] = 1/(500 * 5)
    dummy_train_ds = get_train_dataset(CONFIG)
    assert all(np.isclose(dummy_train_ds.targets,
                      expected_train_targets,
                      atol=0.00001))
    return dummy_train_ds


