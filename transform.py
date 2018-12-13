#!/usr/bin/env python
#coding:utf-8

import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import random
"""
    Do the transform to the image whose format is numpy.ndarray
"""


class Compose(object):
    """
        Compose several transforms together.
        Args:
            transforms (list of ''Transform'' object)
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class Normalize(object):
    def __init__(self, mean, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img -= self.mean
        if self.std:
            img /= self.std + 1e-6
        return img
    
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
            Args:
                img : 4d tensor N x H x W x C
        """
        if random.random() < self.p:
            return img[:,:,::-1,:]
        return img
