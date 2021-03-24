# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F

import math
import numpy as numpy


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        # PIL Image
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target
        # from PIL Image to Tensor
        # (3, h, w), from 0-1


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target





class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4038, 0.4547, 0.4815]):
        # mean=[0.4914, 0.4822, 0.4465]
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def erase_per_box(self, img, bbox):
        if random.uniform(0, 1) > self.probability:
            return img

        in_h, in_w = int(bbox[3] - bbox[1]), int(bbox[2] - bbox[0])

        for attempt in range(100):
            area = in_h * in_w
            
            target_area =  random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < in_w and h < in_h:
                x1 = random.randint(0, in_h - h)
                y1 = random.randint(0, in_w - w)
                if img.size()[0] == 3:
                    img[0, int(bbox[1]+x1):int(bbox[1]+x1+h), int(bbox[0]+y1):int(bbox[0]+y1+w)] = self.mean[0]
                    img[1, int(bbox[1]+x1):int(bbox[1]+x1+h), int(bbox[0]+y1):int(bbox[0]+y1+w)] = self.mean[1]
                    img[2, int(bbox[1]+x1):int(bbox[1]+x1+h), int(bbox[0]+y1):int(bbox[0]+y1+w)] = self.mean[2]
                else:
                    img[2, int(bbox[1]+x1):int(bbox[1]+x1+h), int(bbox[0]+y1):int(bbox[0]+y1+w)] = self.mean[0]
                return img

        return img
       
    def __call__(self, img, target):
        assert target.mode == "xyxy"    
        bbox = target.bbox
        for i in range(len(bbox)):
            img = self.erase_per_box(img, bbox[i])

        return img, target


