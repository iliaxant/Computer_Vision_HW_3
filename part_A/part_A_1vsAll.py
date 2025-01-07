from utils import *
import os
import numpy as np

test_folders = ['caltech-101/caltech-101_5_test/accordion', 'caltech-101/caltech-101_5_test/electric_guitar',
                'caltech-101/caltech-101_5_test/grand_piano', 'caltech-101/caltech-101_5_test/mandolin',
                'caltech-101/caltech-101_5_test/metronome']

bow_descs = np.load('index.npy').astype(np.float32)

img_paths = np.load('paths.npy')

# TRAINING

# Train SVM
print('Training SVMs...')

svm_accordion = cv.ml.SVM_create()
svm_accordion.setType(cv.ml.SVM_C_SVC)
svm_accordion.setKernel(cv.ml.SVM_LINEAR)
svm_accordion.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))  # Max iterations and accuracy

svm_electric_guitar = cv.ml.SVM_create()
svm_electric_guitar.setType(cv.ml.SVM_C_SVC)
svm_electric_guitar.setKernel(cv.ml.SVM_LINEAR)
svm_electric_guitar.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))  # Max iterations and accuracy

svm_grand_piano = cv.ml.SVM_create()
svm_grand_piano.setType(cv.ml.SVM_C_SVC)
svm_grand_piano.setKernel(cv.ml.SVM_LINEAR)
svm_grand_piano.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))  # Max iterations and accuracy

svm_mandolin = cv.ml.SVM_create()
svm_mandolin.setType(cv.ml.SVM_C_SVC)
svm_mandolin.setKernel(cv.ml.SVM_LINEAR)
svm_mandolin.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))  # Max iterations and accuracy

svm_metronome = cv.ml.SVM_create()
svm_metronome.setType(cv.ml.SVM_C_SVC)
svm_metronome.setKernel(cv.ml.SVM_LINEAR)
svm_metronome.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))  # Max iterations and accuracy

labels = []
for p in img_paths:
    if 'caltech-101/caltech-101_5_train/accordion' in p:
        labels.append(1)
    else:
        labels.append(0)

labels = np.array(labels, np.int32)
svm_accordion.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)
svm_accordion.save('svm_accordion')

labels = []
for p in img_paths:
    if 'caltech-101/caltech-101_5_train/electric_guitar' in p:
        labels.append(1)
    else:
        labels.append(0)

labels = np.array(labels, np.int32)
svm_electric_guitar.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)
svm_electric_guitar.save('svm_electric_guitar')

labels = []
for p in img_paths:
    if 'caltech-101/caltech-101_5_train/grand_piano' in p:
        labels.append(1)
    else:
        labels.append(0)

labels = np.array(labels, np.int32)
svm_grand_piano.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)
svm_grand_piano.save('svm_grand_piano')

labels = []
for p in img_paths:
    if 'caltech-101/caltech-101_5_train/mandolin' in p:
        labels.append(1)
    else:
        labels.append(0)

labels = np.array(labels, np.int32)
svm_mandolin.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)
svm_mandolin.save('svm_mandolin')

labels = []
for p in img_paths:
    if 'caltech-101/caltech-101_5_train/metronome' in p:
        labels.append(1)
    else:
        labels.append(0)

labels = np.array(labels, np.int32)
svm_metronome.trainAuto(bow_descs, cv.ml.ROW_SAMPLE, labels)
svm_metronome.save('svm_metronome')

# TESTING
print('Testing System...')
vocabulary = np.load('vocabulary.npy')

# Classification
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

        response_accordion = svm_accordion.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        response_electric_guitar = svm_electric_guitar.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        response_grand_piano = svm_grand_piano.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        response_mandolin = svm_mandolin.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        response_metronome = svm_metronome.predict(bow_desc, flags=cv.ml.STAT_MODEL_RAW_OUTPUT)

        scores = [response_metronome, response_mandolin, response_grand_piano, response_electric_guitar,
                  response_accordion]
        response = scores.index(min(scores))

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
print('# # # # Evaluation of SVM One vs All Classification # # # #')
print()

print('=== [1] Accordions ===')
print('- Correctly Classified: %d' % right_accordion)
print('- Total Accordions: %d' % total_accordion)
success_rate = right_accordion / total_accordion
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print('=== [2] Electric Guitars ===')
print('- Correctly Classified: %d' % right_electric_guitar)
print('- Total Electric Guitars: %d' % total_electric_guitar)
success_rate = right_electric_guitar / total_electric_guitar
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print('=== [3] Grand Pianos ===')
print('- Correctly Classified: %d' % right_grand_piano)
print('- Total Grand Pianos: %d' % total_grand_piano)
success_rate = right_grand_piano / total_grand_piano
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print('=== [4] Mandolins ===')
print('- Correctly Classified: %d' % right_mandolin)
print('- Total Mandolins: %d' % total_mandolin)
success_rate = right_mandolin / total_mandolin
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print('=== [5] Metronomes ===')
print('- Correctly Classified: %d' % right_metronome)
print('- Total Metronomes: %d' % total_metronome)
success_rate = right_metronome / total_metronome
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
print()

print('<<====== Total Accuracy ======>>')
print('- Correctly Classified: %d' % right)
print('- Total Objects: %d' % total)
success_rate = right / total
print('- Success Rate: %.3f => %.1f%%' % (success_rate, 100*success_rate))
