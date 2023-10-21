import tqdm
import os
import glob
import cv2

from model import preprocess

if __name__ == '__main__': 
    paths = glob.glob("datasets/dataset_v2/*.jpg")
    baddies = 0
    for p in tqdm.tqdm(paths):
        try:
            img = cv2.imread(p)
            img = preprocess(img)
        except:
            os.remove(p)
            baddies += 1

