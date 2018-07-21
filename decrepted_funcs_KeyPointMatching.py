# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 04:40:05 2018

Decrepted methos once used in Remote Painting Challenging for TwinOaks.

@author: MRChou
"""

from operator import mul

import numpy
import cv2
from matplotlib import pyplot as plt

TEMPLATE = cv2.imread('FindlandEmblem.jpg', cv2.IMREAD_GRAYSCALE)
RECOGNIZER = TEMPLATE[:90, :90]
SCOREMAP = cv2.imread('FindlandEmblem_ScoreMap.jpg')  # in BGR mode


class CropOp:

    def __init__(self, row_ratio, column_ratio):
        assert 0 < column_ratio <= 1
        assert 0 < row_ratio <= 1
        self._ratio = (row_ratio, column_ratio)
        self._topleft_pixels = None
        self._botleft_pixels = None
        self._topright_pixels = None

    @property
    def topleft_pixels(self):
        return self._topleft_pixels

    @property
    def botleft_pixels(self):
        return self._botleft_pixels

    @property
    def topright_pixels(self):
        return self._topright_pixels

    @property
    def ratio(self):
        return self._ratio

    def crop_topleft(self, img):
        size_row, size_col = map(mul, img.shape, self.ratio)
        self._topleft_pixels = (int(size_row), int(size_col))
        return img[:int(size_row), :int(size_col)]

    def crop_botleft(self, img):
        size_row, size_col = map(mul, img.shape, self.ratio)
        self._botleft_pixels = (int(size_row), int(size_col))
        return img[-int(size_row):img.shape[0], :int(size_col)]

    def crop_topright(self, img):
        size_row, size_col = map(mul, img.shape, self.ratio)
        self._topright_pixels = (int(size_row), int(size_col))
        return img[:int(size_row), -int(size_col):img.shape[1]]


def find_matches(img, recognizer):
    """Find matches between img and recognizer"""
    # Initiate STAR detector
    orb = cv2.ORB_create()

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # find the keypoints with ORB
    kp_rec, des_rec = orb.detectAndCompute(recognizer, None)
    kp_img, des_img = orb.detectAndCompute(img, None)

    # find matches by brute force, sort them in the order of their distance.
    matches = bf.match(des_rec, des_img)
    matches.sort(key=lambda x: x.distance)

    return kp_rec, kp_img, matches


def translate(img, to_left, to_top):
    """Move the img from a given pt by (top, left) to TEMPLATE size"""
    mat = numpy.float32([[1, 0, -1*to_left], [0, 1, -1*to_top]])
    return cv2.warpAffine(img, mat, TEMPLATE.shape[::-1])


def get_vectors_from_matches(kp_query, kp_train, matches):
    matches.sort(key=lambda x: x.distance)

    pts_query = []
    pts_train = []
    for match in matches[:2]:  # Using best 2 matches
        pts_query.append(kp_query[match.queryIdx].pt)
        pts_train.append(kp_train[match.trainIdx].pt)

    vec_query = numpy.subtract(pts_query[1], pts_query[0])
    vec_train = numpy.subtract(pts_train[1], pts_train[0])
    return vec_query, vec_train


def get_angle_and_ratio(vec1, vec2):
    """Return angle(in radian), ratio between two vectors"""
    l1 = numpy.linalg.norm(vec1, ord=2)
    l2 = numpy.linalg.norm(vec2, ord=2)
    sin = numpy.cross(vec1, vec2)/(l1*l2)
    cos = numpy.dot(vec1, vec2)/(l1*l2)
    return numpy.arctan2(sin, cos), l1/l2


def rot_and_resize(img, rad, ratio):
    """Return rotated and resized image according by angle and ratio"""
    h = img.shape[0]
    w = img.shape[1]
    center = (w/2, h/2)
    deg = 180*rad/numpy.pi

    mat = cv2.getRotationMatrix2D(center, deg, ratio)

    # get new image size
    sin = numpy.sin(rad)
    cos = numpy.cos(rad)
    bound_w = h * abs(sin) + w * abs(cos)
    bound_h = h * abs(cos) + w * abs(sin)
    bound_w = int(bound_w * ratio)
    bound_h = int(bound_h * ratio)

    mat[0, 2] += ((bound_w / 2) - center[0])
    mat[1, 2] += ((bound_h / 2) - center[1])

    return cv2.warpAffine(img, mat, (bound_w, bound_h), borderValue=255)


class NoRecogError(Exception):
    """Used when no recognizer found in target image"""
    pass


def cal_score(img, recognizer=RECOGNIZER, max_try=10):
    """Caculate target image score based on recognizer"""

    loop_count = 0
    while True:
        crop_op = CropOp(0.4, 0.4)
        target = crop_op.crop_topleft(img)

        # get key points and the matches
        kp_rec, kp_img, matches = find_matches(target, recognizer=recognizer)

        # get vectors from 2 best matches
        vec_rec, vec_img = get_vectors_from_matches(kp_rec, kp_img, matches)
        angle, ratio = get_angle_and_ratio(vec_rec, vec_img)  # oder of params!
        print('Iter %d: angle= %.5f, ratio= %.5f' % (loop_count, angle, ratio))

        img = rot_and_resize(img, angle, ratio)

        loop_count += 1
        if angle == 0 and ratio == 1:
            break
        elif loop_count == max_try:
            raise NoRecogError('Can not match recognizer Try retaking img.')

    # Now a simple translation movement and crop applied
    pt_in_temp = kp_rec[matches[1].queryIdx].pt
    pt_in_img = kp_img[matches[1].trainIdx].pt
    img = translate(img, *numpy.subtract(pt_in_img, pt_in_temp))
    cv2.imwrite('out.jpg', img)
    return img


def _draw_matching(img, recognizer=TEMPLATE[:90, :90], crop=(0.4, 0.4)):
    crop_op = CropOp(*crop)
    target = crop_op.crop_topleft(img)
    kp_rec, kp_img, matches = find_matches(target, recognizer=recognizer)

    print('Matching Point Img: ', kp_img[matches[0].trainIdx].pt)
    print('Matching Point Rec: ', kp_rec[matches[0].queryIdx].pt)
    plot = cv2.drawMatches(recognizer,
                           kp_rec,
                           img,
                           kp_img,
                           matches[:1],
                           None,
                           flags=2
                           )
    plt.imshow(plot)
    plt.show()
    plt.pause(10)


if __name__ == '__main__':
    TRAIN = cv2.imread('FindlandEmblem_Brushed_rot-18.jpg', cv2.IMREAD_GRAYSCALE)

#    _show_img(TRAIN)
#    _show_img(cal_score(TRAIN))
#    _draw_matching(cv2.imread('out.jpg', cv2.IMREAD_GRAYSCALE))
#    img = numpy.full((900, 740, 3), 0, dtype='uint8')
#    img[:, :, 1] = result # blue
#    img[:, :, 2] = TEMPLATE # red
#    cv2.imshow('', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#    get_score(result)
#    draw_matching(result)
