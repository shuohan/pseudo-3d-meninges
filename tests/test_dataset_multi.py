#!/usr/bin/env python

import random
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from deep_meninges.dataset import create_dataset_multi


random.seed(0)


def test_dataset_multi():
    dirname = 'data_multi'
    num_epochs = 6
    loading_order=['t1w', 't2w', 'outer-mask', 'outer-sdf', 'inner-mask', 'inner-sdf' ]
    dataset = create_dataset_multi(dirname, 16, num_epochs, loading_order=loading_order)
    assert len(dataset._indices) == 2
    assert len(dataset._indices[0]) == 3 * 16
    assert len(dataset._indices[1]) == 3 * 16
    for i in range(num_epochs):
        ind = i % 2
        assert dataset._dataset_index == ind
        for j in range(16):
            len_prev = len(dataset._indices[ind])
            slice_ind = dataset._indices[ind][-1]
            data = dataset[i]
            assert len(data) == 6
            assert 't1w' in data[0].name
            assert 't2w' in data[1].name
            if ind:
                assert 'outer-mask' in data[2].name
                assert 'outer-sdf' in data[3].name
                assert len(data[4]) == 0
                assert len(data[5]) == 0
            else:
                assert 'inner-mask' in data[4].name
                assert 'inner-sdf' in data[5].name
                assert len(data[2]) == 0
                assert len(data[3]) == 0
            assert len(dataset._indices[ind]) == len_prev - 1
        dataset.update()


if __name__ == '__main__':
    test_dataset_multi()
