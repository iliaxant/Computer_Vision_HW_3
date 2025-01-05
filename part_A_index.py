from utils import *
import os
import cv2 as cv
import numpy as np

train_folders = ['caltech-101/caltech-101_5_train/accordion', 'caltech-101/caltech-101_5_train/electric_guitar',
                 'caltech-101/caltech-101_5_train/grand_piano', 'caltech-101/caltech-101_5_train/mandolin',
                 'caltech-101/caltech-101_5_train/metronome']

print('Creating index...')

vocabulary = np.load('vocabulary.npy')

img_paths = []
# train_descs = np.zeros((0, 128))
bow_descs = np.zeros((0, vocabulary.shape[0]))
for folder in train_folders:
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        desc = extract_local_features(path)
        if desc is None:
            continue
        bow_desc = encode_bovw_descriptor(desc, vocabulary)

        img_paths.append(path)
        bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

np.save('index.npy', bow_descs)
np.save('paths', img_paths)
