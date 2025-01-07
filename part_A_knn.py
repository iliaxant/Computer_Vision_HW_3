from utils import *
import os
import cv2 as cv
import numpy as np

train_folders = ['caltech-101/caltech-101_5_train/accordion', 'caltech-101/caltech-101_5_train/electric_guitar',
                 'caltech-101/caltech-101_5_train/grand_piano', 'caltech-101/caltech-101_5_train/mandolin',
                 'caltech-101/caltech-101_5_train/metronome']

test_folders = ['caltech-101/caltech-101_5_test/accordion', 'caltech-101/caltech-101_5_test/electric_guitar',
                'caltech-101/caltech-101_5_test/grand_piano', 'caltech-101/caltech-101_5_test/mandolin',
                'caltech-101/caltech-101_5_test/metronome']

bow_descs = np.load('index.npy').astype(np.float32)

img_paths = np.load('paths.npy')

# TRAINING
labels = []
for p in img_paths:
    if 'caltech-101/caltech-101_5_train/accordion' in p:
        labels.append(4)
    elif 'caltech-101/caltech-101_5_train/electric_guitar' in p:
        labels.append(3)
    elif 'caltech-101/caltech-101_5_train/grand_piano' in p:
        labels.append(2)
    elif 'caltech-101/caltech-101_5_train/mandolin' in p:
        labels.append(1)
    else:
        labels.append(0)

labels = np.array(labels, np.int32)

knn = cv.ml.KNearest_create()
knn.train(bow_descs, cv.ml.ROW_SAMPLE, labels)

# TESTING
vocabulary = np.load('vocabulary.npy')

descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)

right = 0
total = 0

right_accordion = 0
total_accordion = 0

right_electric_guitar = 0
total_electric_guitar = 0

right_grand_piano = 0
total_grand_piano = 0

right_mandolin = 0
total_mandolin = 0

right_metronome = 0
total_metronome = 0

for folder in test_folders:
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)

        img = cv.imread(path)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)

        response, results, neighbours, dist = knn.findNearest(bow_desc, 3)

        total += 1
        if path == os.path.join('caltech-101/caltech-101_5_test/accordion', file):
            flag = 4
            total_accordion += 1
            if response == flag:
                right += 1
                right_accordion += 1
        elif path == os.path.join('caltech-101/caltech-101_5_test/electric_guitar', file):
            flag = 3
            total_electric_guitar += 1
            if response == flag:
                right += 1
                right_electric_guitar += 1
        elif path == os.path.join('caltech-101/caltech-101_5_test/grand_piano', file):
            flag = 2
            total_grand_piano += 1
            if response == flag:
                right += 1
                right_grand_piano += 1
        elif path == os.path.join('caltech-101/caltech-101_5_test/mandolin', file):
            flag = 1
            total_mandolin += 1
            if response == flag:
                right += 1
                right_mandolin += 1
        else:
            flag = 0
            total_metronome += 1
            if response == flag:
                right += 1
                right_metronome += 1

print()
print('# # # # Evaluation of k-NN Classification # # # #')
print()

print()
print('=== [1] Accordions ===')
print('- Correctly Classified: %d' % right_accordion)
print('- Total Accordions: %d' % total_accordion)
success_rate = right_accordion / total_accordion
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print()
print('=== [2] Electric Guitars ===')
print('- Correctly Classified: %d' % right_electric_guitar)
print('- Total Electric Guitars: %d' % total_electric_guitar)
success_rate = right_electric_guitar / total_electric_guitar
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print()
print('=== [3] Grand Pianos ===')
print('- Correctly Classified: %d' % right_grand_piano)
print('- Total Grand Pianos: %d' % total_grand_piano)
success_rate = right_grand_piano / total_grand_piano
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print()
print('=== [4] Mandolins ===')
print('- Correctly Classified: %d' % right_mandolin)
print('- Total Mandolins: %d' % total_mandolin)
success_rate = right_mandolin / total_mandolin
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print()
print('=== [5] Metronomes ===')
print('- Correctly Classified: %d' % right_metronome)
print('- Total Metronomes: %d' % total_metronome)
success_rate = right_metronome / total_metronome
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print()
print('<<====== Total Accuracy ======>>')
print('- Correctly Classified: %d' % right)
print('- Total Objects: %d' % total)
success_rate = right / total
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()
