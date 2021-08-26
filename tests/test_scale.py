#!/usr/bin/env python

import numpy as np
from ct_synth.dataset import Scale
import matplotlib.pyplot as plt


filename = '/iacl/pg20/shuo/work/ct/calamiti/train_data_memmap/MICA-A31_t2w_to-dh-20241-09_data.dat'
data = np.memmap(filename, dtype='float32', shape=(241, 286, 241))
image = data[:, :, 120]


scale = Scale(1.2)

images = [image]
titles = ['orig']
for scales in ([1.2, 1.3], [0.8, 0.7], [1.5, 0.9], [0.5, 2]):
    im = scale._scale_image(image, scales)
    assert im.shape == (241, 286)
    images.append(im)
    titles.append(' '.join(['%g' % s for s in scales]))

plt.figure()
for i, (im, t) in enumerate(zip(images, titles)):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(im, cmap='gray')
    plt.title(t)
plt.show()
