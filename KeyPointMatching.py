# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:46:45 2018

Used in Remote Painting Challenging for TwinOaks.

@author: MRChou
"""

import numpy
import cv2
from matplotlib import pyplot as plt

RECOGNIZER = cv2.imread('Recognizer.jpg', cv2.IMREAD_GRAYSCALE)


def crop_topleft_coner(img, top=0.5, left=0.5):
    """Crop the top-left corner of the original img by the given ratio"""
    size_x, size_y = int(img.shape[0]*left), int(img.shape[1]*top)
    return img[0:size_x, 0:size_y]


def get_vectors_from_matches(kp_query, kp_train, matches):
    matches.sort(key=lambda x: x.distance)

    pts_query = []
    pts_train = []
    for mat in matches[:2]:  # Using best 2 matches
        pts_query.append(kp_query[mat.queryIdx].pt)
        pts_train.append(kp_train[mat.trainIdx].pt)

    vec_query = numpy.subtract(pts_query[1], pts_query[0])
    vec_train = numpy.subtract(pts_train[1], pts_train[0])
    return vec_query, vec_train


def get_angle_and_ratio(vec1, vec2):
    """Return angle(in radian), ratio between two vectors"""
    l1 = numpy.linalg.norm(vec1, ord=2)
    l2 = numpy.linalg.norm(vec2, ord=2)
    dot = numpy.cross(vec1, vec2)/(l1*l2)
    return numpy.arcsin(dot), l1/l2


def rot_and_resize(img):
    """Rotate and resize the image according to RECOGNIZER"""

    # Initiate STAR detector
    orb = cv2.ORB_create()

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # find the keypoints with ORB
    kp_rec, des_rec = orb.detectAndCompute(RECOGNIZER, None)
    kp_img, des_img = orb.detectAndCompute(img, None)

    # find matches by brute force, sort them in the order of their distance.
    matches = bf.match(des_rec, des_img)
    matches.sort(key=lambda x: x.distance)

    # get vectors from 2 best matches
    vec_rec, vec_img = get_vectors_from_matches(kp_rec, kp_img, matches)
    print(vec_rec, vec_img)
    print(get_angle_and_ratio(vec_rec, vec_img))

    # Draw first 10 matches.
    img = cv2.drawMatches(RECOGNIZER, kp_rec,
                          img, kp_img,
                          matches[:5],
                          None,
                          flags=2
                          )

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    TRAIN = cv2.imread('FindlandEmblem_Brushed_Rot36.jpg', cv2.IMREAD_GRAYSCALE)
    TRAIN = crop_topleft_coner(TRAIN)
    rot_and_resize(TRAIN)


