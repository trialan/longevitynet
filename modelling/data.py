import glob
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, random_split, Subset

from longevitynet.modelling.model import FaceAgeDataset
from longevitynet.modelling.config import REPO_DIR
from longevitynet.modelling.utils import set_seed


def get_dataset_dict(config):
    train_dataset = get_train_dataset(config)
    val_dataset, test_dataset = get_val_test_datasets(config)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['BATCH_SIZE'],
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['BATCH_SIZE'],
                                shuffle=False)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config['BATCH_SIZE'],
                                 shuffle=False)
    dataset_dict = {"dataloaders" : {"train": train_dataloader,
                                     "validation": val_dataloader,
                                     "test_dataloader": test_dataloader},
                    "datasets" : {"train": train_dataset,
                                  "validation": val_dataset,
                                  "test": test_dataset}}
    return dataset_dict



def get_train_dataset(config):
    scaling = config["TARGET_SCALER"]
    image_paths = get_train_image_paths(config["DS_VERSION"])
    N_IMAGES = int(len(image_paths) * config["DATA_FRACTION"])
    train_ds = _create_dataset_from_paths(image_paths[:N_IMAGES], scaling)
    print(f"Train Dataset: {len(train_ds)}")
    return train_ds


def get_val_test_datasets(config):
    scaling = config["TARGET_SCALER"]
    image_paths = get_val_test_image_paths(config["DS_VERSION"])
    num_test = int(0.3 * len(image_paths))
    num_val = int(len(image_paths) - num_test)

    test_dataset = _create_dataset_from_paths(image_paths[:num_test],
                                              scaling)
    val_dataset = _create_dataset_from_paths(image_paths[num_test:num_test+num_val],
                                             scaling)
    print(f"Validation Dataset: {len(val_dataset)}")
    print(f"Test Dataset: {len(test_dataset)}")
    return val_dataset, test_dataset


def get_train_image_paths(ds_version):
    image_paths = np.array(glob.glob(f'{REPO_DIR}/datasets/dataset_{ds_version}/*.jpg'))
    return image_paths


def get_val_test_image_paths(ds_version):
    validation_data = f'{REPO_DIR}/datasets/validation_and_test_data_{ds_version}'
    image_paths = np.array(glob.glob(f'{validation_data}/*.jpg'))
    return image_paths


def extract_probs(f):
    p_man = float(f.split('_pman_')[1].split('_')[0].replace('p','.'))
    p_woman = float(f.split('_pwoman_')[1].split(".jpg")[0].split('_')[0].replace('p','.'))
    return {"p_man": p_man, "p_woman": p_woman}


def get_file_data(p):
    name = get_person_name(p)
    probs = extract_probs(p)
    p_man, p_woman = probs['p_man'], probs['p_woman']
    img_date = int(p.split('data:')[-1][:4])
    death_date = int(p.split('death:')[-1][:4])
    birth_date = int(p.split('birth:')[-1][:4])
    data = {"death": death_date,
            "birth": birth_date,
            "age": img_date - birth_date,
            "person_name": name,
            "img_date": img_date,
            "life_expectancy": death_date - img_date,
            "p_man": p_man,
            "p_woman": p_woman}
    assert_name_is_proper(name)
    return data


def get_person_name(p):
    name = p.split("/")[-1].split("_birth")[0]
    for bad_word in ["zoomed_", "flipped_", "gamma_", "gammad_"]:
        if bad_word in name:
            name = name.split(bad_word)[1]
    return name


def _create_dataset_from_paths(image_paths, scaling):
    image_data = [get_file_data(p) for p in image_paths]
    image_dates = [d['img_date'] for d in image_data]
    death_dates = [d['death'] for d in image_data]
    birth_dates = [d['birth'] for d in image_data]
    ages = [d['age'] for d in image_data]
    man_probs = [d['p_man'] for d in image_data]
    woman_probs = [d['p_woman'] for d in image_data]
    life_expectancies = [d['life_expectancy'] for d in image_data]
    assert_life_expectancy_positive(life_expectancies)
    return FaceAgeDataset(image_paths, ages, life_expectancies, man_probs, woman_probs, scaling)


def assert_life_expectancy_positive(life_expectancies):
    bad_ixs = np.where(np.array(life_expectancies) < 0)[0]
    assert sum(bad_ixs) == 0


def assert_name_is_proper(name):
    """ These end up in the name during data augmentation
        Only an issue post-v4."""
    assert "zoomed" not in name
    assert "gamma" not in name
    assert "flipped" not in name


