#!/usr/bin/env python

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from deep_meninges.utils import padcrop


def test_padcrop():
    fn = 'data_multi/inner/OAS30002-d0653_t1w.nii'
    data = nib.load(fn).get_fdata()
    original_shape = data.shape
    target_shape = [256, 288, 288]
    print('orig shape', original_shape, 'target_shape', target_shape)
    print('padded')
    padded = padcrop(data, target_shape, False)
    print('cropped')
    cropped = padcrop(padded, original_shape, False)
    # diff = cropped - data

    # slice_ind = data.shape[2] // 2
    # print(slice_ind)
    # plt.subplot(1, 3, 1)
    # plt.imshow(data[..., slice_ind], cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(cropped[..., slice_ind], cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(diff[..., slice_ind], cmap='gray')
    # plt.show()

    assert np.array_equal(data, cropped)


if __name__ == '__main__':
    test_padcrop()
