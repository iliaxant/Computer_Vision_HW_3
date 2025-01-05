from utils import *
import os
import cv2 as cv
import numpy as np

train_folders = ['caltech-101/caltech-101_5_train/accordion', 'caltech-101/caltech-101_5_train/electric_guitar',
                 'caltech-101/caltech-101_5_train/grand_piano', 'caltech-101/caltech-101_5_train/mandolin',
                 'caltech-101/caltech-101_5_train/metronome']

vocabulary = np.load('vocabulary.npy')

bow_descs = np.load('index.npy')

img_paths = np.load('paths.npy')

# Search
query_img_path = 'images/image_0070.jpg'
query_img = cv.imread(query_img_path)

cv.namedWindow('query', cv.WINDOW_NORMAL)
cv.imshow('query', query_img)

desc = extract_local_features(query_img_path)

bow_desc = encode_bovw_descriptor(desc, vocabulary)

distances = np.sum((bow_desc - bow_descs) ** 2, axis=1)
retrieved_ids = np.argsort(distances)
cv.namedWindow('results', cv.WINDOW_NORMAL)
for id in retrieved_ids.tolist():
    result_img = cv.imread(img_paths[id])
    cv.imshow('results', result_img)
    cv.waitKey(0)
pass

