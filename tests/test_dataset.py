#!/usr/bin/env python 
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from deep_meninges.dataset import create_dataset


def test_dataset():
    dirname = 'data'
    indices = [100, 242, 528, 800, 1053, 1400]
    axes = [0, 1, 2, 0, 1, 2]
    slices = [100, 5, 9, 44, 60, 125]
    names = ['OAS30002-d0653', 'OAS30002-d0653', 'OAS30002-d0653',
             'OAS30003-d0558', 'OAS30003-d0558', 'OAS30003-d0558']

    for target_shape in [None, (288, 288)]:
        if target_shape == (288, 288):
            shapes = [(5, 288, 288)] * 3
            pad = ([slice(1, -1), slice(24, -23)],
                   [slice(24, -23), slice(24, -23)],
                   [slice(24, -23), slice(1, -1)])
        else:
            shapes = [(5, 286, 241), (5, 241, 241), (5, 241, 286)]
            pad = (
                [slice(None), slice(None)],
                [slice(None), slice(None)],
                [slice(None), slice(None)],
            )
        dataset = create_dataset(dirname, target_shape=target_shape,
                                 stack_size=5)
        for ax, ind, n, sind in zip(axes, indices, names, slices):
            t1w, t2w, omask, osdf, imask, isdf = dataset[ind]
            # print(t1w.data.shape, t2w.data.shape, omask.data.shape,
            #       osdf.data.shape, imask.data.shape, isdf.data.shape)

            # plt.subplot(2, 3, 1)
            # plt.imshow(t1w.data[2, ...].T, cmap='gray')
            # plt.subplot(2, 3, 2)
            # plt.imshow(t2w.data[2, ...].T, cmap='gray')
            # plt.subplot(2, 3, 3)
            # plt.imshow(omask.data[2, ...].T, cmap='gray')
            # plt.subplot(2, 3, 4)
            # plt.imshow(osdf.data[2, ...].T, cmap='gray')
            # plt.subplot(2, 3, 5)
            # plt.imshow(imask.data[2, ...].T, cmap='gray')
            # plt.subplot(2, 3, 6)
            # plt.imshow(isdf.data[2, ...].T, cmap='gray')
            # plt.show()

            assert f'{n}_axis-{ax}_slice-{sind}_t1w' == t1w.name
            assert f'{n}_axis-{ax}_slice-{sind}_t2w' == t2w.name
            assert f'{n}_axis-{ax}_slice-{sind}_outer-mask' == omask.name
            assert f'{n}_axis-{ax}_slice-{sind}_outer-sdf' == osdf.name
            assert f'{n}_axis-{ax}_slice-{sind}_inner-mask' == imask.name
            assert f'{n}_axis-{ax}_slice-{sind}_inner-sdf' == isdf.name
            assert t1w.data.shape == shapes[ax]
            assert t2w.data.shape == shapes[ax]
            assert omask.data.shape == shapes[ax]
            assert osdf.data.shape == shapes[ax]
            assert imask.data.shape == shapes[ax]
            assert isdf.data.shape == shapes[ax]

            t1w_fn = Path(dirname, n + '_T1w.nii.gz')
            t2w_fn = Path(dirname, n + '_T2w.nii.gz')
            omask_fn = Path(dirname, n + '_outer-mask.nii.gz')
            osdf_fn = Path(dirname, n + '_outer-sdf.nii.gz')
            imask_fn = Path(dirname, n + '_inner-mask.nii.gz')
            isdf_fn = Path(dirname, n + '_inner-sdf.nii.gz')

            t1w_orig = nib.load(t1w_fn).get_fdata(dtype=np.float32)
            t2w_orig = nib.load(t2w_fn).get_fdata(dtype=np.float32)
            omask_orig = nib.load(omask_fn).get_fdata(dtype=np.float32)
            osdf_orig = nib.load(osdf_fn).get_fdata(dtype=np.float32)
            imask_orig = nib.load(imask_fn).get_fdata(dtype=np.float32)
            isdf_orig = nib.load(isdf_fn).get_fdata(dtype=np.float32)
            if ax == 0:
                t1w_orig = np.flip(t1w_orig[sind : sind + 5, ...], -1)
                t2w_orig = np.flip(t2w_orig[sind : sind + 5, ...], -1)
                omask_orig = np.flip(omask_orig[sind : sind + 5, ...], -1)
                osdf_orig = np.flip(osdf_orig[sind : sind + 5, ...], -1)
                imask_orig = np.flip(imask_orig[sind : sind + 5, ...], -1)
                isdf_orig = np.flip(isdf_orig[sind : sind + 5, ...], -1)

                assert np.array_equal(t1w_orig, t1w.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(t2w_orig, t2w.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(omask_orig, omask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(osdf_orig, osdf.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(imask_orig, imask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(isdf_orig, isdf.data[:, pad[ax][0], pad[ax][1]])

            elif ax == 1:
                t1w_orig = np.flip(np.moveaxis(t1w_orig[:, sind : sind + 5, :], ax, 0), -1)
                t2w_orig = np.flip(np.moveaxis(t2w_orig[:, sind : sind + 5, :], ax, 0), -1)
                omask_orig = np.flip(np.moveaxis(omask_orig[:, sind : sind + 5, :], ax, 0), -1)
                osdf_orig = np.flip(np.moveaxis(osdf_orig[:, sind : sind + 5, :], ax, 0), -1)
                imask_orig = np.flip(np.moveaxis(imask_orig[:, sind : sind + 5, :], ax, 0), -1)
                isdf_orig = np.flip(np.moveaxis(isdf_orig[:, sind : sind + 5, :], ax, 0), -1)

                # print(t2w_orig.shape, t2w.data[:, pad[ax][0], pad[ax][1]].shape)
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[0, ...].sum())
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[1, ...].sum())
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[2, ...].sum())
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[3, ...].sum())
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[4, ...].sum())

                # print(omask.data[:, pad[ax][0], pad[ax][1]].min(), omask.data[:, pad[ax][0], pad[ax][1]].max())

                # plt.figure()
                # plt.imshow(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[2, ...])
                # plt.show()
                # plt.figure()
                # plt.imshow(omask_orig[2, ...])
                # plt.show()

                assert np.array_equal(t1w_orig, t1w.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(t2w_orig, t2w.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(omask_orig, omask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(osdf_orig, osdf.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(imask_orig, imask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(isdf_orig, isdf.data[:, pad[ax][0], pad[ax][1]])

            elif ax == 2:
                t1w_orig = np.moveaxis(t1w_orig[..., sind : sind + 5], ax, 0)
                t2w_orig = np.moveaxis(t2w_orig[..., sind : sind + 5], ax, 0)
                omask_orig = np.moveaxis(omask_orig[..., sind : sind + 5], ax, 0)
                osdf_orig = np.moveaxis(osdf_orig[..., sind : sind + 5], ax, 0)
                imask_orig = np.moveaxis(imask_orig[..., sind : sind + 5], ax, 0)
                isdf_orig = np.moveaxis(isdf_orig[..., sind : sind + 5], ax, 0)

                assert np.array_equal(t1w_orig, t1w.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(t2w_orig, t2w.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(omask_orig, omask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(osdf_orig, osdf.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(imask_orig, imask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(isdf_orig, isdf.data[:, pad[ax][0], pad[ax][1]])


def test_dataset_no_t1w():
    dirname = 'data'
    indices = [100, 242, 528, 800, 1053, 1400]
    axes = [0, 1, 2, 0, 1, 2]
    slices = [100, 5, 9, 44, 60, 125]
    names = ['OAS30002-d0653', 'OAS30002-d0653', 'OAS30002-d0653',
             'OAS30003-d0558', 'OAS30003-d0558', 'OAS30003-d0558']

    for target_shape in [None, (288, 288)]:
        if target_shape == (288, 288):
            shapes = [(5, 288, 288)] * 3
            pad = ([slice(1, -1), slice(24, -23)],
                   [slice(24, -23), slice(24, -23)],
                   [slice(24, -23), slice(1, -1)])
        else:
            shapes = [(5, 286, 241), (5, 241, 241), (5, 241, 286)]
            pad = (
                [slice(None), slice(None)],
                [slice(None), slice(None)],
                [slice(None), slice(None)],
            )
        dataset = create_dataset(dirname, target_shape=target_shape,
                                 skip=['t1w'], stack_size=5)
        for ax, ind, n, sind in zip(axes, indices, names, slices):
            t2w, omask, osdf, imask, isdf = dataset[ind]

            # plt.subplot(2, 3, 2)
            # plt.imshow(t2w.data[2, ...].T, cmap='gray')
            # plt.subplot(2, 3, 3)
            # plt.imshow(omask.data[2, ...].T, cmap='gray')
            # plt.subplot(2, 3, 4)
            # plt.imshow(osdf.data[2, ...].T, cmap='gray')
            # plt.subplot(2, 3, 5)
            # plt.imshow(imask.data[2, ...].T, cmap='gray')
            # plt.subplot(2, 3, 6)
            # plt.imshow(isdf.data[2, ...].T, cmap='gray')
            # plt.show()

            assert f'{n}_axis-{ax}_slice-{sind}_t2w' == t2w.name
            assert f'{n}_axis-{ax}_slice-{sind}_outer-mask' == omask.name
            assert f'{n}_axis-{ax}_slice-{sind}_outer-sdf' == osdf.name
            assert f'{n}_axis-{ax}_slice-{sind}_inner-mask' == imask.name
            assert f'{n}_axis-{ax}_slice-{sind}_inner-sdf' == isdf.name
            assert t2w.data.shape == shapes[ax]
            assert omask.data.shape == shapes[ax]
            assert osdf.data.shape == shapes[ax]
            assert imask.data.shape == shapes[ax]
            assert isdf.data.shape == shapes[ax]

            t2w_fn = Path(dirname, n + '_T2w.nii.gz')
            omask_fn = Path(dirname, n + '_outer-mask.nii.gz')
            osdf_fn = Path(dirname, n + '_outer-sdf.nii.gz')
            imask_fn = Path(dirname, n + '_inner-mask.nii.gz')
            isdf_fn = Path(dirname, n + '_inner-sdf.nii.gz')

            t2w_orig = nib.load(t2w_fn).get_fdata(dtype=np.float32)
            omask_orig = nib.load(omask_fn).get_fdata(dtype=np.float32)
            osdf_orig = nib.load(osdf_fn).get_fdata(dtype=np.float32)
            imask_orig = nib.load(imask_fn).get_fdata(dtype=np.float32)
            isdf_orig = nib.load(isdf_fn).get_fdata(dtype=np.float32)
            if ax == 0:
                t2w_orig = np.flip(t2w_orig[sind : sind + 5, ...], -1)
                omask_orig = np.flip(omask_orig[sind : sind + 5, ...], -1)
                osdf_orig = np.flip(osdf_orig[sind : sind + 5, ...], -1)
                imask_orig = np.flip(imask_orig[sind : sind + 5, ...], -1)
                isdf_orig = np.flip(isdf_orig[sind : sind + 5, ...], -1)
                assert np.array_equal(t2w_orig, t2w.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(omask_orig, omask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(osdf_orig, osdf.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(imask_orig, imask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(isdf_orig, isdf.data[:, pad[ax][0], pad[ax][1]])

            elif ax == 1:
                t2w_orig = np.flip(np.moveaxis(t2w_orig[:, sind : sind + 5, :], ax, 0), -1)
                omask_orig = np.flip(np.moveaxis(omask_orig[:, sind : sind + 5, :], ax, 0), -1)
                osdf_orig = np.flip(np.moveaxis(osdf_orig[:, sind : sind + 5, :], ax, 0), -1)
                imask_orig = np.flip(np.moveaxis(imask_orig[:, sind : sind + 5, :], ax, 0), -1)
                isdf_orig = np.flip(np.moveaxis(isdf_orig[:, sind : sind + 5, :], ax, 0), -1)

                # print(t2w_orig.shape, t2w.data[:, pad[ax][0], pad[ax][1]].shape)
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[0, ...].sum())
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[1, ...].sum())
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[2, ...].sum())
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[3, ...].sum())
                # print(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[4, ...].sum())

                # print(omask.data[:, pad[ax][0], pad[ax][1]].min(), omask.data[:, pad[ax][0], pad[ax][1]].max())

                # plt.figure()
                # plt.imshow(np.abs(omask_orig - omask.data[:, pad[ax][0], pad[ax][1]])[2, ...])
                # plt.show()
                # plt.figure()
                # plt.imshow(omask_orig[2, ...])
                # plt.show()

                assert np.array_equal(t2w_orig, t2w.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(omask_orig, omask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(osdf_orig, osdf.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(imask_orig, imask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(isdf_orig, isdf.data[:, pad[ax][0], pad[ax][1]])

            elif ax == 2:
                t2w_orig = np.moveaxis(t2w_orig[..., sind : sind + 5], ax, 0)
                omask_orig = np.moveaxis(omask_orig[..., sind : sind + 5], ax, 0)
                osdf_orig = np.moveaxis(osdf_orig[..., sind : sind + 5], ax, 0)
                imask_orig = np.moveaxis(imask_orig[..., sind : sind + 5], ax, 0)
                isdf_orig = np.moveaxis(isdf_orig[..., sind : sind + 5], ax, 0)
                assert np.array_equal(t2w_orig, t2w.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(omask_orig, omask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(osdf_orig, osdf.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(imask_orig, imask.data[:, pad[ax][0], pad[ax][1]])
                assert np.array_equal(isdf_orig, isdf.data[:, pad[ax][0], pad[ax][1]])


if __name__ == '__main__':
    test_dataset()
    test_dataset_no_t1w()
