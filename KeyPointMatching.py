# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 17:46:45 2018

Used in Remote Painting Challenging for TwinOaks.

@author: MRChou
"""

from operator import mul
from collections import namedtuple

import numpy
import cv2

TEMPLATE = cv2.imread('./Images/FindlandEmblem.jpg', cv2.IMREAD_GRAYSCALE)

SCOREMAP = cv2.imread('./Images/FindlandEmblem_ScoreMap.png')  # in BGR mode
SCORECUT = (81, 85)

# Positive score region is red (channel 2 = Red)
POS_SCORE = numpy.where(SCOREMAP[..., 2] > 200, 0, 255).astype(numpy.uint8)

# Negative score region is green (channel 1 = Red)
NEG_SCORE = numpy.where(SCOREMAP[..., 1] > 200, 0, 255).astype(numpy.uint8)


def show_img(imgs, names=None, **kargs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    if names is None:
        names = [str(index) for index in range(len(imgs))]
    elif not isinstance(names, list):
        names = [names]

    for img, name in zip(imgs, names):
        if img.shape[1] > 500:
            ratio = 500/img.shape[1]
            img = cv2.resize(img, None,  fx=ratio, fy=ratio)
        cv2.imshow(name, img, **kargs)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def check_matching(target, template, matching_pixels):

    # convert grayscale image to color
    temp = numpy.zeros((*template.shape, 3), numpy.uint8)
    tar = numpy.zeros((*target.shape, 3), numpy.uint8)
    for i in range(3):
        temp[:, :, i] = template
        tar[:, :, i] = target

    # drawing circles
    temp_circle_size = int(5*max(temp.shape)/900)
    tar_circle_size = int(5*max(tar.shape)/900)

    cv2.circle(temp, matching_pixels[0].temp_pixel[::-1],
               temp_circle_size, (0, 0, 255), -1)  # red
    cv2.circle(temp, matching_pixels[1].temp_pixel[::-1],
               temp_circle_size, (0, 255, 0), -1)  # green
    cv2.circle(temp, matching_pixels[2].temp_pixel[::-1],
               temp_circle_size, (255, 0, 0), -1)  # blue

    cv2.circle(tar, matching_pixels[0].target_pixel[::-1],
               tar_circle_size, (0, 0, 255), -1)
    cv2.circle(tar, matching_pixels[1].target_pixel[::-1],
               tar_circle_size, (0, 255, 0), -1)
    cv2.circle(tar, matching_pixels[2].target_pixel[::-1],
               tar_circle_size, (255, 0, 0), -1)

    show_img(names=['template_pts', 'target_pts'], imgs=[temp, tar])
    return None


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


_match = namedtuple('Matching_Pixel', (['target_pixel', 'temp_pixel']))


class ImgMatcher:
    """For performing 3-corner match between a template and a target image"""

    def __init__(self, target_img, template=TEMPLATE, crop=CropOp(0.4, 0.4)):
        self.target = target_img
        self.template = template
        self.crop = crop
        self.target_corners = [crop.crop_botleft(target_img),
                               crop.crop_topleft(target_img),
                               crop.crop_topright(target_img)
                               ]
        self.temp_corners = [template[-90:, :90],
                             template[:90, :90],
                             template[:90, -90:]
                             ]
        self.crop_pixels = [(target_img.shape[0]-crop.botleft_pixels[0], 0),
                            (0, 0),
                            (0, target_img.shape[1]-crop.topright_pixels[1])
                            ]
        self.temp_pixels = [(template.shape[0]-90, 0),
                            (0, 0),
                            (0, template.shape[1]-90)
                            ]

    def three_pt_match(self):
        matching_pixels = []
        for tar_cor, temp_cor, tar_pix, temp_pix in zip(self.target_corners,
                                                        self.temp_corners,
                                                        self.crop_pixels,
                                                        self.temp_pixels):
            # use cropped image to find macth
            kp_temp, kp_tar, matches = find_matches(
                    img=tar_cor, recognizer=temp_cor
                    )

            # get the single best match
            matches.sort(key=lambda x: x.distance)
            match = matches[0]

            # add cropped pixel values back, to get the originial pixel
            def int_add(a, b):
                return int(a+b)

            # Keypoint.pt is a tuple of (dim_x, dim_y) == (col, row)!
            target_pixel = tuple(map(int_add,
                                     kp_tar[match.trainIdx].pt[::-1],
                                     tar_pix))
            temp_pixel = tuple(map(int_add,
                                   kp_temp[match.queryIdx].pt[::-1],
                                   temp_pix))

            matching_pixels.append(
                    _match(target_pixel=target_pixel,
                           temp_pixel=temp_pixel)
                    )
        return matching_pixels

    def get_affine_transform_mat(self):
        match_pts = self.three_pt_match()

        # A pt should be a list of (dim_x, dim_y) == (col, row)!
        pt_tar = numpy.array([[*(pt.target_pixel[::-1])] for pt in match_pts],
                             dtype=numpy.float32)
        pt_temp = numpy.array([[*(pt.temp_pixel[::-1])] for pt in match_pts],
                              dtype=numpy.float32)

        mat = cv2.getAffineTransform(pt_tar, pt_temp)
        return mat

    def recursive_affine_transform(self, max_try=10):
        # the 2x3 invariant matrix
        const_mat = numpy.array([[1., 0., 0.], [0., 1., 0]])
        matcher = ImgMatcher(target_img=self.target,
                             template=self.template,
                             crop=self.crop)

        for _ in range(max_try):
            try:
                mat = matcher.get_affine_transform_mat()
            except cv2.error:
                raise cv2.error('Matching pattern not found, try bigger crop!')

            img = cv2.warpAffine(matcher.target, mat, (10000, 10000))
            img = img[:matcher.template.shape[0], :matcher.template.shape[1]]
            matcher = ImgMatcher(target_img=img,
                                 template=matcher.template,
                                 crop=matcher.crop)

            if numpy.array_equal(mat, const_mat):
                print('Transform Finished. Use .target to get the image')
                break
            else:
                # use large blank image to hold transformed image, then cut
                print('Itering... Mat=')
                print(mat)

        return matcher.target


def simple_clahe(img):
    """Use contrast-enhansed image and simple mean-cut to threshold image."""
    target = img.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(50, 50))
    target = clahe.apply(target)
    return target


def threshold(img, thres_min=100, thres_max=200, split=20):
    """Convolve target with a max-pooling, to get regional threshold value,
    with pooling kernel size=stride= 1/20 image size (i.e. non-overlapping)"""
    target = img.copy()

    # apply min-max scale
    target = target + (255 - target.max())

    width = int(target.shape[0]/split)
    height = int(target.shape[1]/split)

    thres = []
    for row_iter in range(split):
        row_i, row_f = (row_iter * width, (row_iter+1) * width)
        for col_iter in range(split):
            col_i, col_f = (col_iter * height, (col_iter+1) * height)

            # min = most white pixel
            thres.append(target[row_i:row_f, col_i:col_f].max())

    # thres is darkest out of whitest
    global_thres = min(thres)

    if global_thres < thres_min:
        global_thres = thres_min
    elif global_thres > thres_max:
        global_thres = thres_max

    # if pixel < thres (i.e. darker than thres), let it be black
    target = numpy.where(target < global_thres, 0, 255).astype(numpy.uint8)
    return target


def get_drawing(img):
    target = img.copy()

    # first thresholding
    target = threshold(target)

    # then mask out scoring boarders
    mask = numpy.zeros(target.shape, dtype=bool)
    mask[SCORECUT[0]:, SCORECUT[1]:] = True
    target[~mask] = 255

    return target


def get_score(img):
    target = img.copy()

    target = get_drawing(target)

    pos = numpy.where(numpy.logical_and(target == 0, POS_SCORE == 0), 0, 255)
    neg = numpy.where(numpy.logical_and(target == 0, NEG_SCORE == 0), 0, 255)

    score = numpy.count_nonzero(pos == 0) - numpy.count_nonzero(neg == 0)
    return pos, neg, score


if __name__ == '__main__':
    pass
