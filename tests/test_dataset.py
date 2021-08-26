#!/usr/bin/env python

import matplotlib.pyplot as plt
from ct_synth.dataset import create_dataset


def test_dataset():
    dirname = '../data_test'
    axes = [0, 1, 2, 0, 1, 2]
    indices = [100, 242, 528, 800, 3372, 3600]
    names = ['MICA-A38', 'MICA-A38', 'MICA-A38', 'MICA-A41', 'MICA-A48', 'MICA-A48']
    slices = [100, 1, 1, 32, 59, 1]

    for target_shape in [None, (288, 288)]:
        if target_shape == (288, 288):
            shapes = [(288, 288)] * 3
        else:
            shapes = [(286, 241), (241, 241), (241, 286)]
        dataset = create_dataset(dirname, target_shape=target_shape)
        for ax, ind, n, sind in zip(axes, indices, names, slices):
            t1w, t2w, ct, mask = dataset[ind]

            # plt.subplot(1, 4, 1)
            # plt.imshow(t12w.data[0, ...].T, cmap='gray')
            # plt.subplot(1, 4, 2)
            # plt.imshow(t12w.data[1, ...].T, cmap='gray')
            # plt.subplot(1, 4, 3)
            # plt.imshow(ct.data[0, ...].T, cmap='gray')
            # plt.subplot(1, 4, 4)
            # plt.imshow(mask.data[0, ...].T, cmap='gray')
            # plt.show()

            assert f'{n}_axis-{ax}_slice-{sind}_t1w' == t1w.name
            assert f'{n}_axis-{ax}_slice-{sind}_t2w' == t2w.name
            assert f'{n}_axis-{ax}_slice-{sind}_ct' == ct.name
            assert f'{n}_axis-{ax}_slice-{sind}_mask' == mask.name
            assert t1w.data.shape == (1, *shapes[ax])
            assert t2w.data.shape == (1, *shapes[ax])
            assert ct.data.shape == (1, *shapes[ax])
            assert mask.data.shape == (1, *shapes[ax])


def test_dataset_no_t1w():
    dirname = '../data_test'
    axes = [0, 1, 2, 0, 1, 2]
    indices = [100, 242, 528, 800, 3372, 3600]
    names = ['MICA-A38', 'MICA-A38', 'MICA-A38', 'MICA-A41', 'MICA-A48', 'MICA-A48']
    slices = [100, 1, 1, 32, 59, 1]

    for target_shape in [None, (288, 288)]:
        if target_shape == (288, 288):
            shapes = [(288, 288)] * 3
        else:
            shapes = [(286, 241), (241, 241), (241, 286)]
        dataset = create_dataset(dirname, target_shape=target_shape, skip=['t1w'])
        for ax, ind, n, sind in zip(axes, indices, names, slices):
            t2w, ct, mask = dataset[ind]

            # plt.subplot(1, 4, 1)
            # plt.imshow(t12w.data[0, ...].T, cmap='gray')
            # plt.subplot(1, 4, 2)
            # plt.imshow(t12w.data[1, ...].T, cmap='gray')
            # plt.subplot(1, 4, 3)
            # plt.imshow(ct.data[0, ...].T, cmap='gray')
            # plt.subplot(1, 4, 4)
            # plt.imshow(mask.data[0, ...].T, cmap='gray')
            # plt.show()

            assert f'{n}_axis-{ax}_slice-{sind}_t2w' == t2w.name
            assert f'{n}_axis-{ax}_slice-{sind}_ct' == ct.name
            assert f'{n}_axis-{ax}_slice-{sind}_mask' == mask.name
            assert t2w.data.shape == (1, *shapes[ax])
            assert ct.data.shape == (1, *shapes[ax])
            assert mask.data.shape == (1, *shapes[ax])


if __name__ == '__main__':
    test_dataset()
    test_dataset_no_t1w()
