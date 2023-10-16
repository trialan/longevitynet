import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Subset

from life_expectancy.modelling.model import FaceAgeDataset

TEST_SET_RATIO = 0.2

def get_dataloaders(ds_version, batch_size, seed):
    dataset = generate_dataset(ds_version)
    train_dataset, test_dataset = _get_train_test_split(dataset, seed)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
    return train_dataloader, test_dataloader


def generate_dataset(ds_version):
    image_paths = np.array(glob.glob(f'life_expectancy/datasets/dataset_{ds_version}/*.jpg'))
    image_dates = np.array([int(p.split('data:')[-1][:-4]) for p in image_paths])
    death_dates = np.array([int(p.split('death:')[-1][:4]) for p in image_paths])
    birth_dates = np.array([int(p.split('birth:')[-1][:4]) for p in image_paths])
    ages = np.array([img_date - birth_date for img_date, birth_date in zip(image_dates, birth_dates)])
    life_expectancies = np.array([death - date for death, date in zip(death_dates, image_dates)])
    good_ixs = np.where(life_expectancies > 0)[0]
    dataset = FaceAgeDataset(image_paths[good_ixs], ages[good_ixs], life_expectancies[good_ixs])
    return dataset


def _get_train_test_split(dataset, seed):
    num_test = int(TEST_SET_RATIO * len(dataset))
    num_train = len(dataset) - num_test
    train_dataset, test_dataset = random_split(dataset, [num_train, num_test],
                                               generator=torch.Generator().manual_seed(seed))
    return train_dataset, test_dataset


