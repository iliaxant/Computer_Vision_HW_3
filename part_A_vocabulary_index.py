from utils import *
import os
import cv2 as cv
import numpy as np

train_folders = ['caltech-101/caltech-101_5_train/accordion', 'caltech-101/caltech-101_5_train/electric_guitar',
                 'caltech-101/caltech-101_5_train/grand_piano', 'caltech-101/caltech-101_5_train/mandolin',
                 'caltech-101/caltech-101_5_train/metronome']


# Extract Database
print('Extracting features...')
train_descs = np.zeros((0, 128))
for folder in train_folders:
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        desc = extract_local_features(path)
        if desc is None:
            continue
        train_descs = np.concatenate((train_descs, desc), axis=0)

# Create vocabulary
print('Creating vocabulary...')
term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
trainer = cv.BOWKMeansTrainer(50, term_crit, 1, cv.KMEANS_PP_CENTERS)
vocabulary = trainer.cluster(train_descs.astype(np.float32))

np.save('vocabulary.npy', vocabulary)

print('Creating index...')
# Classification
descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

img_paths = []
bow_descs = np.zeros((0, vocabulary.shape[0]))
for folder in train_folders:
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)

        img = cv.imread(path)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)

        img_paths.append(path)
        bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

np.save('index.npy', bow_descs)
np.save('paths', img_paths)
