import re
import random
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset as Dataset_
from pathlib import Path
from cv2 import resize, INTER_CUBIC
from collections import OrderedDict, deque

from ptxl.utils import NamedData

from .utils import padcrop


random.seed(16784)


def create_dataset(
        dirname,
        target_shape=(288, 288),
        transform=None,
        shuffle_once=False,
        memmap=False,
        stack_size=1,
        loading_order=[],
    ):
    GI = GroupImagesMemmap if memmap else GroupImages
    images = GI(dirname, loading_order).group()
    subjects = list()
    for subname, filenames in images.items():
        SD = SubjectDataMemmap if memmap else SubjectData
        subject = SD(
            subname,
            filenames,
            im_types=loading_order,
            target_shape=target_shape,
            transform=transform,
            shuffle_once=shuffle_once,
            stack_size=stack_size,
        )
        subjects.append(subject)
    dataset = Dataset(subjects)
    return dataset


def create_dataset_multi(
        dirname,
        batch_size,
        num_slices_per_epoch,
        num_epochs,
        target_shape=(288, 288),
        transform=None,
        shuffle_once=False,
        memmap=False,
        stack_size=1,
        loading_order=[],
        random_indices=True,
    ):
    group_images_cls = GroupImagesMemmap if memmap else GroupImages
    group_images = GroupImagesMulti(dirname, loading_order, group_images_cls)
    all_images = group_images.group()

    datasets = list()
    for images in all_images:
        subjects = list()
        for subname, filenames in images.items():
            SD = SubjectDataMemmap if memmap else SubjectData
            subject = SD(
                subname,
                filenames,
                im_types=loading_order,
                target_shape=target_shape,
                transform=transform,
                shuffle_once=shuffle_once,
                stack_size=stack_size,
            )
            subjects.append(subject)
        datasets.append(Dataset(subjects))
    ds = DatasetMulti(
        datasets,
        batch_size,
        num_slices_per_epoch,
        num_epochs,
        random_indices=random_indices,
    )
    return ds


class GroupImages_:
    def group(self):
        raise NotImplementedError


class GroupImages(GroupImages_):
    def __init__(self, dirname, loading_order=[]):
        self.dirname = dirname
        self.loading_order = loading_order
        self._re_pattern = '(' + '|'.join(self.loading_order) + ')'

    def group(self):
        images = OrderedDict()
        for image_fn in self._find_images():
            names = self._parse_name(image_fn)
            subname, im_type = names[:2]
            if subname not in images:
                images[subname] = OrderedDict()
            if im_type not in images[subname]:
                images[subname][im_type] = list()
            images[subname][im_type].append(image_fn)
        return images

    def _find_images(self):
        def has_im_type(fn):
            return re.search(self._re_pattern, str(fn))
        filenames = Path(self.dirname).glob('*.nii*')
        filenames = list(filter(has_im_type, filenames))
        filenames = self._sort_images(filenames)
        return filenames

    def _sort_images(self, filenames):
        keys = {k: v for v, k in enumerate(self.loading_order)}
        def get_sort_key(fn):
            return keys[re.search(self._re_pattern, str(fn)).group()]
        filenames = sorted(filenames, key=get_sort_key)
        def get_subj_name(fn):
            return fn.name.split('_')[0]
        filenames = sorted(filenames, key=get_subj_name)
        return filenames

    def _parse_name(self, image_fn):
        return re.sub(r'\.nii(\.gz)*$', '', Path(image_fn).name).split('_')


class GroupImagesMemmap(GroupImages):
    def _find_images(self):
        return sorted(Path(self.dirname).glob('*.dat'))

    def _parse_name(self, image_fn):
        return re.sub(r'\.dat$', '', Path(image_fn).name).split('_')


class GroupImagesMulti(GroupImages_):
    def __init__(self, dirname, loading_order=[],
                 group_images_cls=GroupImages):
        self.dirname = dirname
        self.loading_order = loading_order
        self.group_images = [
            group_images_cls(subdir, loading_order)
            for subdir in sorted(Path(dirname).glob('*'))
        ]

    def group(self):
        images = list()
        for gi in self.group_images:
            images.append(gi.group())
        return images


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

    def update(self):
        pass


class DatasetMulti(Dataset_):
    def __init__(self, datasets, batch_size, num_slices_per_epoch, num_epochs,
                 random_indices=True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_slices_per_epoch = num_slices_per_epoch
        self.num_epochs = num_epochs
        self.random_indices = random_indices
        self._num_datasets = len(self.datasets)
        self._calc_num_slices_per_dataset()

        if self.random_indices:
            self._sample_indices()
        else:
            self._get_consecutive_indices()

        self._dataset_order = np.arange(self._num_datasets)

    def _calc_num_slices_per_dataset(self):
        self._num_slices_per_dataset \
            = self.num_slices_per_epoch * self.num_epochs

    def _sample_indices(self):
        self._indices = list()
        for i, ds in enumerate(self.datasets):
            indices = random.choices(
                range(len(ds)),
                k=self._num_slices_per_dataset
            )
            indices = deque(indices)
            self._indices.append(indices)

    def _get_consecutive_indices(self):
        self._indices = list()
        for i, ds in enumerate(self.datasets):
            indices = np.linspace(0, len(ds), self.num_slices_per_epoch + 2)
            indices = np.round(indices[1:-1]).astype(int).tolist()
            indices = indices * self.num_epochs
            random.shuffle(indices)
            indices = deque(indices)
            self._indices.append(indices)

    def __len__(self):
        return self.num_slices_per_epoch * self._num_datasets

    def __getitem__(self, index):
        ds_ind = self._split_index(index)
        dataset = self.datasets[ds_ind]
        indices = self._indices[ds_ind]
        index = indices.pop()
        return dataset[index]

    def _split_index(self, index):
        num_batches_per_ds = self.num_slices_per_epoch // self.batch_size
        shape = (num_batches_per_ds, self._num_datasets, self.batch_size)
        batch_ind, ds_ind, sample_ind = np.unravel_index(index, shape)
        ds_ind = self._dataset_order[ds_ind]
        return ds_ind

    def update(self):
        self._dataset_order = np.roll(self._dataset_order, 1)


class SubjectData:

    def __init__(
            self, name, filenames,
            im_types=None,
            target_shape=None,
            transform=None,
            shuffle_once=False,
            stack_size=1,
        ):
        self.name = name
        self.target_shape = target_shape
        self.transform = transform
        self.im_types = im_types
        if self.im_types is None:
            self.im_types = list(filenames.keys())
        self.shuffle_once = shuffle_once
        self.stack_size = stack_size
        self._create_images(filenames)
        self._check_im_shapes()

    @property
    def shape(self):
        return list(self._images.values())[0][0].shape

    def _create_images(self, filenames):
        self._images = OrderedDict()
        for im_type in self.im_types:
            self._images[im_type] = list()

        for im_type, fns in filenames.items():
            for fn in fns:
                nifti = nib.load(fn)
                contrast = self._get_contrast(fn)
                im_slices = ImageSlicesDataobj(
                    nifti, contrast,
                    stack_size=self.stack_size,
                    target_shape=self.target_shape
                )
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
        if not self._images[im_type]:
            return []

        if self.shuffle_once:
            image_slices = self._images[im_type][0]
        else:
            image_slices = random.choice(self._images[im_type])
        im_slice = image_slices[index]
        name = '_'.join([self.name, im_slice.name, im_type])
        return NamedData(name=name, data=im_slice.data)

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
                contrast = self._get_contrast(fn)
                im_slices = ImageSlicesMemmap(
                    fp, contrast,
                    stack_size=self.stack_size,
                    target_shape=self.target_shape
                )
                self._images[im_type].append(im_slices)
            if self.shuffle_once:
                random.shuffle(self._images[im_type])

    def _get_contrast(self, fn):
        result = re.sub(r'_data\.dat$', '', Path(fn).name)
        result = result.split('_')
        result = result[2] if len(result) > 2 else ''
        return result


class ImageSlices:
    def __init__(self, nifti, name, stack_size=1, target_shape=None):
        self.nifti = nifti
        self.name = name
        self.target_shape = target_shape
        self.stack_size = stack_size
        assert self.stack_size % 2 == 1
        self._num_cumsum = np.cumsum(self.shape)

    @property
    def shape(self):
        return tuple(s - self.stack_size + 1 for s in self.nifti.shape)

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
        image_slice = np.moveaxis(image_slice, axis, 0)
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
        result[axis] = slice(index, index + self.stack_size)
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
                if len(image) == 0:
                    results.append(image)
                else:
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
                if len(image) == 0:
                    results.append(image)
                else:
                    assert image.data.ndim == 3
                    data = self._scale_images(image.data, scales)
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

    def _scale_images(self, data, scales):
        results = list()
        for image in data:
            d = self._scale_image(image, scales)
            results.append(d)
        return np.stack(results, axis=0)

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
