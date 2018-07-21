# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 06:02:01 2018

Script for evaluating score of remote painting activity.

@author: MRChou
"""

import os

import numpy
import cv2

from KeyPointMatching import TEMPLATE
from KeyPointMatching import ImgMatcher, CropOp, show_img, check_matching
from KeyPointMatching import get_drawing, get_score


def test_match():
    in_path = './quick_test/'
    out_path = './quick_test_out/'
    test_imgs = [in_path + img for img in os.listdir(in_path) if img.startswith('TestImage')]

    for test_img in test_imgs:
        target = cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)

        matcher = ImgMatcher(target_img=target, crop=CropOp(0.4, 0.4))
        show_img(names=os.path.basename(test_img), imgs=matcher.target)

        # view crop
        show_img(names=['crop_botleft', 'crop_topleft', 'crop_topright'],
                 imgs=matcher.target_corners)

        # check matchings
        check_matching(target=matcher.target,
                       template=matcher.template,
                       matching_pixels=matcher.three_pt_match())

        # get final result
        try:
            result = matcher.recursive_affine_transform()
        except cv2.error as err:
            print(err)
        else:
            show_img(names='final result', imgs=result)

            assert result.shape == TEMPLATE.shape
            cv2.imwrite(out_path + os.path.basename(test_img), result)


def end_to_end_score(imgfile, out_path):
    target = cv2.imread(imgfile, cv2.IMREAD_GRAYSCALE)
    print('Processing {}'.format(os.path.basename(imgfile)))

    matcher = ImgMatcher(target_img=target, crop=CropOp(0.4, 0.4))
    try:
        result = matcher.recursive_affine_transform()
    except cv2.error as err:
        print(err)
    else:
        assert result.shape == TEMPLATE.shape
        print('Finish affine transform.')

        draw_region = get_drawing(result)

        pos, neg, score = get_score(result)
        score_region = numpy.full((*TEMPLATE.shape, 3), 255, dtype=numpy.uint8)
        score_region[..., 0] = pos  # turn off blue
        score_region[..., 1] = pos  # turn off green
        score_region[..., 2] = neg  # turn off red

        cv2.imwrite(os.path.join(out_path, 'raw_input.jpg'), target)
        cv2.imwrite(os.path.join(out_path, 'affine_transform.jpg'), result)
        cv2.imwrite(os.path.join(out_path, 'draw_region.jpg'), draw_region)
        cv2.imwrite(os.path.join(out_path, 'scores_{}.jpg'.format(score)), score_region)

        show_img(names=['Raw input {}'.format(os.path.basename(imgfile)),
                        'Apply affine transform',
                        'Binarize Drawing',
                        'Final Score = {}'.format(score)],
                 imgs=[target,
                       result,
                       draw_region,
                       score_region]
                 )


if __name__ == '__main__':
    in_path = './Team_Test/0_pics/'
    out_path = './Team_Test/'
    imgfiles = [in_path + file for file in os.listdir(in_path) if 'Regular' in file]

    for imgfile in imgfiles:
        outputdir = out_path + os.path.basename(imgfile).strip('.jpg')
        os.mkdir(outputdir)
        end_to_end_score(imgfile, outputdir)

    print('Done!')
