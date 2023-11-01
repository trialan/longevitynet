import pytest

from longevitynet.modelling.config import CONFIG
from longevitynet.modelling.data import (get_train_image_paths,
                                            get_val_test_image_paths,
                                            get_file_data)


def test_no_overlap_between_train_and_test():
    """ Don't test overlap in files, test overlap in people names """
    ds_version = CONFIG["DS_VERSION"]
    train_image_paths = get_train_image_paths(ds_version)
    test_image_paths = get_val_test_image_paths()
    people_in_train = [get_file_data(f)['person_name'] for f
                       in train_image_paths]
    people_in_test = [get_file_data(f)['person_name'] for f
                       in test_image_paths]
    overlap = set(people_in_train).intersection(people_in_test)
    assert len(overlap) == 0


def test_getting_file_data():
    example_paths = get_val_test_image_paths()[:4]
    expected_data = [{"person_name": "Carsten Eggers",
                      "birth": 1957, "death": 2021,
                      "img_date": 2000, "life_expectancy": 21},
                     {"person_name": "Béla Éless",
                      "birth": 1940, "death": 2020,
                      "img_date": 2014, "life_expectancy": 6},
                     {"person_name": "Claude Gingras",
                      "birth": 1931, "death": 2018,
                      "img_date": 2017, "life_expectancy": 1},
                     {"person_name": "Franz Muheim",
                      "birth": 1931, "death": 2020,
                      "img_date": 1982, "life_expectancy": 32}]

    for i, path in enumerate(example_paths):
        data = get_file_data(path)
        assert data['person_name'] == expected_data[i]['person_name']
        assert data['death'] == expected_data[i]['death']
        assert data['birth'] == expected_data[i]['birth']

