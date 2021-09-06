import torch
import numpy as np


def padcrop(image, target_shape, use_channels=True):
    shape = image.shape[1:] if use_channels else image.shape
    bbox = calc_padcrop_bbox(shape, target_shape)
    lpads = [max(0, 0 - b.start) for b in bbox]
    rpads = [max(0, b.stop - s) for s, b in zip(shape, bbox)]
    pad_width = tuple(zip(lpads, rpads))
    c_bbox = tuple(slice(b.start + l, b.stop + l) for b, l in zip(bbox, lpads))
    if use_channels:
        pad_width = ((0, 0), ) + pad_width
        c_bbox = (..., ) + c_bbox
    padded_image = np.pad(image, pad_width, mode='edge')
    cropped_image = padded_image[c_bbox]
    return cropped_image


def calc_padcrop_bbox(source_shape, target_shape):
    bbox = list()
    for ss, ts in zip(source_shape, target_shape):
        diff = ss - ts
        left = np.abs(diff) // 2 * np.sign(diff)
        right = left + ts
        bbox.append(slice(left, right))
    return tuple(bbox)
