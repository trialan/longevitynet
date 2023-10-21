import numpy as np
import cv2
import dlib
from PIL import Image


base_path = "/Users/thomasrialan/Documents/code/longevity_project/life_expectancy/datasets/dataset_v3/"


def dlib_count_faces(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img_array = np.array(img)
    detector = dlib.get_frontal_face_detector()
    faces = detector(img_array)
    return len(faces)


def haar_cascade_count_faces(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces)


if __name__ == '__main__':
    two_faces = ["Adriana_Roel_birth:1934_death:2022_data:1959.jpg",
                 "Ahlem_Belhadj_birth:1964_death:2023_data:2018.jpg",
                 "Aleksandr_Churilin_birth:1946_death:2021_data:2008.jpg",
                 "Alfonso_Quaranta_birth:1936_death:2023_data:2011.jpg",
                 "Allan_Jay_birth:1931_death:2023_data:1960.jpg",
                 "Allin_Vlasenko_birth:1938_death:2021_data:2016.jpg",
                 "André_Cognat_birth:1938_death:2021_data:1979.jpg"]

    single_face = ["Albin_Molnár_birth:1935_death:2022_data:2003.jpg",
                   "Aleksandr_Sloboda_birth:1920_death:2022_data:2016.jpg",
                   "Aled_Roberts_birth:1962_death:2022_data:2011.jpg",
                   "Allan_Wood_birth:1943_death:2022_data:1964.jpg",
                   "Aloys_Jousten_birth:1937_death:2021_data:2006.jpg",
                   "Amancio_Amaro_birth:1939_death:2023_data:1971.jpg",
                   "Anatoly_Grigoryev_birth:1943_death:2023_data:2015.jpg"]


    print("Not single face expected")
    for f in two_faces:
        print(haar_cascade_count_faces(base_path + f))

    print("Single face expected")
    for f in single_face:
        print(haar_cascade_count_faces(base_path + f))


