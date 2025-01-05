import os
import cv2 as cv
import numpy as np

sift = cv.xfeatures2d_SIFT.create()


def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc


def encode_bovw_descriptor(desc, vocabulary):
    bow_desc = np.zeros((1, vocabulary.shape[0]))
    for d in range(desc.shape[0]):
        distances = np.sum((desc[d, :] - vocabulary) ** 2, axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1

    # Normalization
    if np.sum(bow_desc) > 0:
        bow_desc = bow_desc / np.sum(bow_desc)
    return bow_desc
