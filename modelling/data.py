import glob
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, random_split, Subset

from life_expectancy.modelling.model import FaceAgeDataset

VAL_SET_RATIO = 0.2
TEST_SET_RATIO = 0.05

def get_dataloaders(dataset, config):
    train_dataset, val_dataset, test_dataset = _get_splits(dataset, config['SEED'])

    num_samples = int(config['TRAIN_SIZE_FRACTION'] * len(train_dataset))
    indices = torch.randperm(len(train_dataset))[:num_samples]
    train_dataset = Subset(train_dataset, indices)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['BATCH_SIZE'],
                                  shuffle=True)
    val_dataloader   = DataLoader(val_dataset,
                                 batch_size=config['BATCH_SIZE'],
                                 shuffle=False)
    test_dataloader  = DataLoader(test_dataset,
                                 batch_size=config['BATCH_SIZE'],
                                 shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


def get_dataset(config):
    ds_version = config["DS_VERSION"]
    scaling = config["TARGET_SCALER"]


    image_paths = np.array(glob.glob(f'life_expectancy/datasets/dataset_{ds_version}/*.jpg'))
    image_dates = np.array([int(p.split('data:')[-1][:-4]) for p in image_paths])
    death_dates = np.array([int(p.split('death:')[-1][:4]) for p in image_paths])
    birth_dates = np.array([int(p.split('birth:')[-1][:4]) for p in image_paths])
    ages = np.array([img_date - birth_date for img_date, birth_date in zip(image_dates, birth_dates)])
    life_expectancies = np.array([death - date for death, date in zip(death_dates, image_dates)])
    good_ixs = np.where(life_expectancies > 0)[0]
    dataset = FaceAgeDataset(image_paths[good_ixs], ages[good_ixs], life_expectancies[good_ixs], scaling)
    return dataset


def _get_splits(dataset, seed):
    num_test = int(TEST_SET_RATIO * len(dataset))
    num_val = int(VAL_SET_RATIO * len(dataset))
    num_train = len(dataset) - num_test - num_val
    train_dataset, temp_dataset = random_split(dataset, [num_train, len(dataset) - num_train],
                                               generator=torch.Generator().manual_seed(seed))
    val_dataset, test_dataset = random_split(temp_dataset, [num_val, num_test],
                                             generator=torch.Generator().manual_seed(seed))
    return train_dataset, val_dataset, test_dataset
