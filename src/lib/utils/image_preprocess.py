import os
import sys
import pathlib
import time
import collections
import itertools
import shutil
import pickle
import inspect
import json
import subprocess
import logging
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
from tqdm import tqdm


def preprocess_with_info(img, image_wh):
    h, w, c = img.shape
    w_r = image_wh[0] / w
    h_r = image_wh[1] / h

    if h_r > w_r:
        resize_ratio = w_r
        resize_other_length = int(h * resize_ratio)
        resize_wh = (image_wh[0], resize_other_length)
        diff = image_wh[1] - resize_other_length
        if diff % 2 == 0:
            pad_length = diff // 2
            pad = [[pad_length, pad_length], [0, 0], [0, 0]]
            resize_wh_range = [[0, image_wh[0]], [
                pad_length, image_wh[1]-pad_length]]

        else:
            pad_length = diff // 2
            pad = [[pad_length+1, pad_length], [0, 0], [0, 0]]
            resize_wh_range = [[0, image_wh[0]], [
                pad_length+1, image_wh[1]-pad_length]]

    else:
        resize_ratio = h_r
        resize_other_length = int(w*resize_ratio)
        resize_wh = (resize_other_length, image_wh[1])
        diff = image_wh[0] - resize_other_length
        if diff % 2 == 0:
            pad_length = diff // 2
            pad = [[0, 0], [pad_length, pad_length], [0, 0]]
            resize_wh_range = [
                [pad_length, image_wh[0]-pad_length], [0, image_wh[1]]]

        else:
            pad_length = diff // 2
            pad = [[0, 0], [pad_length+1, pad_length], [0, 0]]
            resize_wh_range = [
                [pad_length+1, image_wh[0]-pad_length], [0, image_wh[1]]]

    img = cv2.resize(img, resize_wh)
    img = np.pad(img, pad, mode="constant", constant_values=0)
    meta = {
        "pad": pad,
        "resize_ratio": resize_ratio,
        "resize_wh": resize_wh,
        "org_wh": (w, h),
        "resize_wh_range": resize_wh_range
    }

    return img, meta


def denormalize_images(tensor, mean, std):
    if isinstance(mean, list):
        mean = torch.FloatTensor(mean)[None, :, None, None]
        std = torch.FloatTensor(std)[None, :, None, None]
    tensor = tensor * std + mean
    tensor = tensor * 255

    tensor = tensor.permute(0, 2, 3, 1).numpy()
    tensor = tensor.astype(np.uint8)
    tensor = np.ascontiguousarray(tensor)

    return tensor


def expand_bbox(bbox, image_wh, is_xywh, expansion_rate):
    if not is_xywh:
        x1, y1 = bbox[:2]
        w = bbox[2] - x1
        h = bbox[3] - y1

    else:
        x1, y1, w, h = bbox

    x1_n = x1 - w * (expansion_rate - 1) / 2
    x2_n = x1_n + w * expansion_rate
    y1_n = y1 - h * (expansion_rate - 1) / 2
    y2_n = y1_n + h * expansion_rate

    x1_n, x2_n, y1_n, y2_n = map(int, (x1_n, x2_n, y1_n, y2_n))

    x1_n = max(0, x1_n)
    x2_n = min(image_wh[0], x2_n)

    y1_n = max(0, y1_n)
    y2_n = min(image_wh[1], y2_n)

    return [x1_n, y1_n, x2_n, y2_n]


def generate_random(bbox, is_xywh):
    if is_xywh:
        bbox_wh = bbox[2:]

        x1, y1 = bbox[:2]
        x2 = bbox[0] + bbox[2]
        y2 = bbox[1] + bbox[3]

    else:
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_wh = [bbox_w, bbox_h]

        x1, y1, x2, y2 = bbox

    range_w = min(2, bbox_wh[0]*0.1)
    random_x1, random_x2 = np.random.randint(-range_w, range_w, size=2)
    range_h = min(2, bbox_wh[1]*0.1)
    random_y1, random_y2 = np.random.randint(-range_h, range_h, size=2)
    x1 = x1 - random_x1
    x2 = x2 - random_x2
    y1 = y1 - random_y1
    y2 = y2 - random_y2
    x1, x2, y1, y2 = map(int, (x1, x2, y1, y2))

    x1 = max(0, x1)
    x2 = max(x1+1, x2)
    y1 = max(0, y1)
    y2 = max(y1+1, y2)

    return [x1, y1, x2, y2]


def crop_image_by_bbox(img, bbox_x1y1x2y2, random_clip=False, expansion_rate=1.5):
    image_wh = img.shape[:2][::-1]
    x1, y1, x2, y2 = bbox_x1y1x2y2
    x1, y1, x2, y2 = expand_bbox(
        bbox_x1y1x2y2,
        image_wh=image_wh,
        expansion_rate=expansion_rate,
        is_xywh=False
    )

    if random_clip:
        x1, y1, x2, y2 = generate_random(
            bbox=bbox_x1y1x2y2,
            is_xywh=False
        )

    img = img[y1: y2, x1: x2]

    return img
