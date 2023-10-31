"""
The best model should have high performance:
    - over the whole age distribution
    - across both genders
-> If my dataset is imbalanced, it will reduce performance.
"""
import glob
import matplotlib.pyplot as plt
import numpy as np

from life_expectancy import REPO_DIR
from life_expectancy.modelling.data import get_file_data

train_data = glob.glob(f'{REPO_DIR}/datasets/dataset_v5/*.jpg')
test_data = glob.glob(f'{REPO_DIR}/datasets/validation_and_test_data_v5/*.jpg')


if __name__ == '__main__':
    data = train_data
    metadata = [get_file_data(f) for f in data]

    ages = np.array([d['age'] for d in metadata])
    p_man = np.array([d['p_man'] for d in metadata])
    p_woman = np.array([d['p_woman'] for d in metadata])

    fraction_men = np.mean(p_man > 0.5)
    fraction_women = np.mean(p_woman > 0.5)
    print(f"Fraction of men in dataset: {fraction_men:.2f}")
    print(f"Fraction of women in dataset: {fraction_women:.2f}")

    input("Hit any key to continue")

    plt.hist(ages, bins=20, edgecolor='k')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig("dataset_v5_age_distribution.png")

