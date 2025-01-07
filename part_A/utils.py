import cv2 as cv

sift = cv.xfeatures2d_SIFT.create()


def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc
