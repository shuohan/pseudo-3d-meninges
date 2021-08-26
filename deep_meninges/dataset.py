import re
import random
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset as Dataset_
from pathlib import Path
from cv2 import resize, INTER_CUBIC

from ptxl.utils import NamedData

from .utils import padcrop


random.seed(16784)


def create_dataset(dirname, target_shape=(288, 288), transform=None, skip=[],
                   shuffle_once=False, memmap=False):
    GI = GroupImagesMemmap if memmap else GroupImages
    images = GI(dirname, skip).group()
    subjects = list()
    for subname, filenames in images.items():
        SD = SubjectDataMemmap if memmap else SubjectData
        subject = SD(subname, filenames, target_shape, transform, shuffle_once)
        subjects.append(subject)
    dataset = Dataset(subjects)
    return dataset


class GroupImages:
    def __init__(self, dirname, skip=[]):
        self.dirname = dirname
        self.skip = skip

    def group(self):
        images = dict()
        for image_fn in self._find_images():
            names = self._parse_name(image_fn)
            subname, im_type = names[:2]
            if subname not in images:
                images[subname] = dict()
            if im_type not in self.skip:
                if im_type not in images[subname]:
                    images[subname][im_type] = list()
                images[subname][im_type].append(image_fn)
        return images

    def _find_images(self):
        return sorted(Path(self.dirname).glob('*.nii*'))

    def _parse_name(self, image_fn):
        return re.sub(r'\.nii(\.gz)*$', '', Path(image_fn).name).split('_')


class GroupImagesMemmap(GroupImages):
    def _find_images(self):
        return sorted(Path(self.dirname).glob('*.dat'))

    def _parse_name(self, image_fn):
        return re.sub(r'\.dat$', '', Path(image_fn).name).split('_')


class Dataset(Dataset_):
    def __init__(self, subjects):
        self.subjects = subjects
        num_slices = [len(sub) for sub in self.subjects]
        self._num_slices_cumsum = np.cumsum(num_slices)

    def __len__(self):
        return self._num_slices_cumsum[-1]

    def __getitem__(self, index):
        sub_ind, slice_ind = self._split_index(index)
        slices = self.subjects[sub_ind][slice_ind]
        return slices

    def _split_index(self, index):
        sub_ind = np.searchsorted(self._num_slices_cumsum, index, side='right')
        offset = 0 if sub_ind == 0 else self._num_slices_cumsum[sub_ind - 1]
        slice_ind = index - offset
        return sub_ind, slice_ind


class SubjectData:

    def __init__(self, name, filenames, target_shape=None, transform=None,
                 shuffle_once=False):
        self.name = name
        self.target_shape = target_shape
        self.transform = transform
        self.im_types = sorted(filenames.keys(), key=self._get_sort_keys)
        self.shuffle_once = shuffle_once
        self._check_im_types(filenames)
        self._create_images(filenames)
        self._check_im_shapes()

    def _get_sort_keys(self, name):
        # use dict in case one of the contrast is missing
        sort_keys = {'t1w': 0, 't2w': 1, 'ct': 2, 'mask': 3}
        return sort_keys[name]

    def _check_im_types(self, filenames):
        assert sorted(self.im_types) == sorted(list(filenames.keys()))

    def _create_images(self, filenames):
        self._images = dict()
        for im_type, fns in filenames.items():
            self._images[im_type] = list()
            for fn in fns:
                nifti = nib.load(fn)
                # print('Load', self, im_type, fn)
                contrast = self._get_contrast(fn)
                im_slices = ImageSlicesDataobj(nifti, contrast, self.target_shape)
                self._images[im_type].append(im_slices)
            if self.shuffle_once:
                random.shuffle(self._images[im_type])

    def _get_contrast(self, fn):
        result = re.sub(r'\.nii(\.gz)*$', '', Path(fn).name)
        result = result.split('_')
        result = result[2] if len(result) > 2 else ''
        return result

    def _check_im_shapes(self):
        shapes = [im.shape for ims in self._images.values() for im in ims]
        assert len(set(shapes)) == 1
 
    def __getitem__(self, index):
        data = [self._get_slices(imt, index) for imt in self.im_types]
        if self.transform:
            data = self.transform(*data)
        return data

    def _get_slices(self, im_type, index):
        if self.shuffle_once:
            image_slices = self._images[im_type][0]
            # print('Valid', self.name, im_type, image_slices.name)
        else:
            image_slices = random.choice(self._images[im_type])

        im_slice = image_slices[index]
        name = '_'.join([self.name, im_slice.name, im_type])
        # print('Selected', image_slices, name)
        return NamedData(name=name, data=im_slice.data[None, ...])

    def __len__(self):
        key = list(self._images.keys())[0]
        return len(self._images[key][0])


class SubjectDataMemmap(SubjectData):

    def _create_images(self, filenames):
        self._images = dict()
        for im_type, fns in filenames.items():
            self._images[im_type] = list()
            for fn in fns:
                shape_fn = re.sub(r'_data\.dat$', '_shape.npy', str(fn))
                dtype_fn = re.sub(r'_data\.dat$', '_dtype.txt', str(fn))
                with open(dtype_fn) as f:
                    dtype = f.readline()
                shape = tuple(np.load(shape_fn).tolist())
                fp = np.memmap(fn, dtype=dtype, mode='r', shape=shape)
                # print('Load', self, im_type, fn)
                contrast = self._get_contrast(fn)
                im_slices = ImageSlicesMemmap(fp, contrast, self.target_shape)
                self._images[im_type].append(im_slices)
            if self.shuffle_once:
                random.shuffle(self._images[im_type])

    def _get_contrast(self, fn):
        result = re.sub(r'_data\.dat$', '', Path(fn).name)
        result = result.split('_')
        result = result[2] if len(result) > 2 else ''
        return result


class ImageSlices:
    def __init__(self, nifti, name, target_shape=None):
        self.nifti = nifti
        self.name = name
        self.target_shape = target_shape
        self._num_cumsum = np.cumsum(self.nifti.shape)

    @property
    def shape(self):
        return self.nifti.shape

    def __getitem__(self, index):
        self._check_index(index)
        axis, slice_ind = self._split_index(index)
        image_slice = self.extract_slice(axis, slice_ind)
        return image_slice

    def _split_index(self, index):
        axis = np.searchsorted(self._num_cumsum, index, side='right')
        offset = 0 if axis == 0 else self._num_cumsum[axis - 1]
        index = index - offset
        return axis, index

    def extract_slice(self, axis, index):
        indexing = self._calc_indexing(axis, index)
        image_slice = self._extract_slice(indexing)
        if axis in [0, 1]: # top bottom swap
            image_slice = np.flip(image_slice, -1)
        if self.target_shape:
            image_slice = padcrop(image_slice, self.target_shape)
        name = self._get_name(axis, index)
        image_slice = NamedData(name=name, data=image_slice)
        return image_slice

    def _extract_slice(self, indexing):
        image = self.nifti.get_fdata(dtype=np.float32)
        image_slice = image[indexing]
        return image_slice

    def _get_name(self, axis, slice_ind):
        name = [f'axis-{axis}', f'slice-{slice_ind}']
        if self.name:
            name.insert(0, self.name)
        return '_'.join(name)

    def _calc_indexing(self, axis, index):
        result = [slice(None)] * self.nifti.ndim
        result[axis] = index
        return tuple(result)

    def __len__(self):
        return self._num_cumsum[-1]

    def _check_index(self, index):
        assert index >= 0 and index < len(self)


class ImageSlicesDataobj(ImageSlices):
    def _extract_slice(self, indexing):
        image = self.nifti.dataobj
        image_slice = image[indexing].astype(np.float32)
        return image_slice


class ImageSlicesMemmap(ImageSlices):
    def _extract_slice(self, indexing):
        return self.nifti[indexing].copy()


class FlipLR:
    def __call__(self, *images):
        results = images
        need_to_flip = np.random.uniform(0, 1) < 0.5
        need_to_flip = ('axis-0' not in images[0].name) and need_to_flip
        if need_to_flip:
            results = list()
            for image in images:
                assert image.data.ndim == 3
                data = np.flip(image.data, axis=1).copy()
                name = '_'.join([image.name, 'flip'])
                results.append(NamedData(name=name, data=data))
        return results


class Scale:
    def __init__(self, scale=1.2):
        self.scale = scale
        assert self.scale >= 1

    def __call__(self, *images):
        results = images
        need_to_scale = np.random.uniform(0, 1) < 0.5
        if need_to_scale:
            results = list()
            scales = self._sample_scales()
            for image in images:
                assert image.data.ndim == 3
                data = self._scale_image(image.data[0, ...], scales)[None, ...]
                name = self._get_name(image.name, scales)
                results.append(NamedData(name=name, data=data))
        return results

    def _sample_scales(self):
        flip_signs = np.random.choice([False, True], 2)
        scales = np.random.uniform(1, self.scale, 2)
        result = list()
        for f, s in zip(flip_signs, scales):
            result.append(1 / s if f else s)
        return result

    def _get_name(self, image_name, dxy):
        dxy = [('%.2f' % d).replace('.', 'p') for d in dxy]
        dxy = '-'.join(dxy)
        name = '_'.join([image_name, 'scale-%s' % dxy])
        return name

    def _scale_image(self, data, scales):
        target_shape = [int(round(data.shape[1] * scales[1])),
                        int(round(data.shape[0] * scales[0]))]
        result = resize(data, target_shape, interpolation=INTER_CUBIC)
        result = self._padcrop(result, data.shape)
        return result

    def _padcrop(self, image, target_shape, pad_mode='edge'):
        diffs =[ss - ts for ss, ts in zip(image.shape, target_shape)]
        ldiffs = [d // 2 for d in diffs]
        rdiffs = [d - ld for d, ld in zip(diffs, ldiffs)]

        lpads = [max(0, -ld) for ld in ldiffs]
        rpads = [max(0, -rd) for rd in rdiffs]
        pad_width = list(zip(lpads, rpads))
        pad_width = tuple(pad_width)
        padded_image = np.pad(image, pad_width, mode=pad_mode)

        lcrops = [max(0, ld) for ld in ldiffs]
        rcrops = [max(0, rd) for rd in rdiffs]
        crop_bbox = [slice(l, s - r)
                     for l, r, s in zip(lcrops, rcrops, padded_image.shape)]
        crop_bbox = tuple(crop_bbox)
        cropped_image = padded_image[crop_bbox]

        return cropped_image


class Compose:
    """Composes several transforms together.

    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
