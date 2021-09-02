import torch
import numpy as np


def padcrop(image, target_shape):
    bbox = calc_padcrop_bbox(image.shape[1:], target_shape)
    lpads = [max(0, 0 - b.start) for b in bbox]
    rpads = [max(0, b.stop - s) for s, b in zip(image.shape[1:], bbox)]
    pad_width = ((0, 0), ) + tuple(zip(lpads, rpads))
    padded_image = np.pad(image, pad_width, mode='edge')
    c_bbox = [slice(b.start + l, b.stop + l) for b, l in zip(bbox, lpads)]
    cropped_image = padded_image[(..., ) + tuple(c_bbox)]
    return cropped_image


def calc_padcrop_bbox(source_shape, target_shape):
    bbox = list()
    for ss, ts in zip(source_shape, target_shape):
        left = (ss - ts) // 2
        right = left + ts
        bbox.append(slice(left, right))
    return tuple(bbox)
