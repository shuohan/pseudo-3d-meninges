#!/usr/bin/env python

import numpy as np
from ct_synth.utils import padcrop


def test_padcrop():
    image = np.hstack([np.zeros([10, 5]), np.ones([10, 10])])
    result = padcrop(image, (5, 20))
    ref = np.hstack([np.zeros((5, 8)), np.ones((5, 12))])
    assert np.array_equal(result, ref)


if __name__ == '__main__':
    test_padcrop()
