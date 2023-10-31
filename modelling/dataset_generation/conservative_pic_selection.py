from PIL import Image
from tqdm import tqdm
import glob
import shutil
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from multilingual_wikidata.count_faces import dlib_count_faces, haar_cascade_count_faces


files = glob.glob("manual_cleanup/*.jpg")
assert len(files) > 16e3


if __name__ == '__main__':
    errors = 0
    for file in tqdm(files):
        try:
            n_faces_dlib = dlib_count_faces(file)
            n_faces_haar = haar_cascade_count_faces(file)
        except:
            errors += 1
            print(f"Errors: {errors}")
            print(f"File: {file}")
            continue

        ensemble = n_faces_haar + n_faces_dlib

        if ensemble == 2:
            shutil.move(file, "clean_v1/")
