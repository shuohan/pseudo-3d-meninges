import torch
import re
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage.measurements import label as find_cc
from copy import deepcopy

from .network import UNet
from .dataset import ImageSlices
from .utils import padcrop


class TesterBoth:
    def __init__(self, args):
        self.args = args
        self._parse_args()
        self._load_model()
        self._create_testers()
        self._create_combine_images()

    def _parse_args(self):
        Path(self.args.output_dir).mkdir(exist_ok=True, parents=True)

        if self.args.use_cuda:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

    def _load_model(self):
        in_channels = 2 if self.args.input_data_mode == 't1w_t2w' else 1
        out_channels = 2 if self.args.output_data_mode == 'ct_mask' else 1
        model = UNet(in_channels, out_channels, 5, self.args.num_channels)
        self._model = model.cuda() if self.args.use_cuda else model
        self._cp = torch.load(self.args.checkpoint, map_location=self._device)
        self._model.load_state_dict(self._cp['model_state_dict'])

    def _create_testers(self):
        input_slices = list()
        self._nifti = list()
        target_shape = self.args.target_shape
        if 't1w' in self.args.input_data_mode:
            t1w_obj = nib.load(self.args.t1w)
            t1w_slices = ImageSlices(t1w_obj, '', target_shape=target_shape)
            input_slices.append(t1w_slices)
            self._nifti.append(t1w_obj)
        if 't2w' in self.args.input_data_mode:
            t2w_obj = nib.load(self.args.t2w)
            t2w_slices = ImageSlices(t2w_obj, '', target_shape=target_shape)
            input_slices.append(t2w_slices)
            self._nifti.append(t2w_obj)

        self._testers = list()
        for axis in range(3):
            self._testers.append(self._create_tester(input_slices, axis))

    def _create_tester(self, input_slices, axis):
        return TesterBoth_(self._model, input_slices, CombineSlices(axis), self.args)

    def _create_combine_images(self):
        if self.args.combine == 'median':
            self._combine_images = CalcMedian()
        elif self.args.combine == 'mean':
            self._combine_images = CalcMean()

    def test(self):
        cts = list()
        masks = list()
        for axis, tester in enumerate(self._testers):
            ct, mask = tester.test()
            ct = padcrop(ct, self._nifti[0].shape)
            mask = padcrop(mask, self._nifti[0].shape)
            self._save_image(ct, 'ct_axis-%d' % axis)
            self._save_image(mask, 'mask_axis-%d' % axis)
            cts.append(ct)
            masks.append(mask)
        ct = self._combine_images(cts)
        mask = self._combine_images(masks)
        mask = self._cleanup_mask(mask)
        self._save_image(ct, 'ct')
        self._save_image(mask > 0.5, 'mask')

    def _save_image(self, im, name):
        filename = Path(self.args.t1w).name
        filename = re.sub(r'\.nii(\.gz)*$', '', filename)
        filename = '_'.join([filename, name])
        filename = Path(self.args.output_dir, filename).with_suffix('.nii.gz')
        obj = nib.Nifti1Image(im, self._nifti[0].affine, self._nifti[0].header)
        obj.to_filename(filename)

    def _cleanup_mask(self, mask):
        mask = mask > 0.5
        labels, num_labels = find_cc(mask)
        counts = np.bincount(labels.flatten())
        for l in np.argsort(counts)[::-1]:
            fg = labels == l
            if np.array_equal(np.unique(mask[fg]), np.array([True])):
                break
        return fg


class TesterCT(TesterBoth):
    def _create_tester(self, input_slices, axis):
        return TesterCT_(self._model, input_slices, CombineSlices(axis), self.args)

    def test(self):
        cts = list()
        for axis, tester in enumerate(self._testers):
            ct = tester.test()
            ct = padcrop(ct, self._nifti[0].shape)
            self._save_image(ct, 'ct_axis-%d' % axis)
            cts.append(ct)
        ct = self._combine_images(cts)
        self._save_image(ct, 'ct')


class TesterMask(TesterBoth):
    def _create_tester(self, input_slices, axis):
        return TesterMask_(self._model, input_slices, CombineSlices(axis), self.args)

    def test(self):
        masks = list()
        levelsets = list()
        for axis, tester in enumerate(self._testers):
            mask, ls = tester.test()
            mask = padcrop(mask, self._nifti[0].shape)
            ls = padcrop(ls, self._nifti[0].shape)
            self._save_image(mask, 'mask_axis-%d' % axis)
            self._save_image(ls, 'ls_axis-%d' % axis)
            masks.append(mask)
            levelsets.append(ls)
        mask = self._combine_images(masks)
        mask = self._cleanup_mask(mask)
        ls = self._combine_images(levelsets)
        self._save_image(mask > 0.5, 'mask')
        self._save_image(ls, 'ls')
        fusion = self._fuse_mask_ls(mask, ls)
        self._save_image(fusion, 'fusion')

    def _fuse_mask_ls(self, mask, ls):
        inv_mask = np.logical_not(mask)
        result_ls = deepcopy(ls)
        result_ls[mask] = np.clip(ls[mask], -self.args.max_ls_value, 0)
        result_ls[inv_mask] = np.clip(ls[inv_mask], 0, self.args.max_ls_value)
        return result_ls

class CalcMedian:
    def __call__(self, images):
        return np.median(np.stack(images, axis=0), axis=0)


class CalcMean:
    def __call__(self, images):
        return np.mean(np.stack(images, axis=0), axis=0)


class CombineSlices:
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, images):
        if self.axis in [0, 1]: # top bottom swap
            images = [np.flip(im, -1) for im in images]
        images = np.concatenate(images, axis=0).squeeze()
        if self.axis == 1:
            images = np.transpose(images, (1, 0, 2))
        elif self.axis == 2:
            images = np.transpose(images, (1, 2, 0))
        return images


class TesterBoth_:
    def __init__(self, model, input_slices, combine_slices, args):
        self.model = model
        self.input_slices = input_slices
        self.combine_slices = combine_slices
        self.args = args

        if self.args.use_cuda:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        self._axis = self.combine_slices.axis
        self._check_shape()

    def _check_shape(self):
        nums = [slices.shape[self._axis] for slices in self.input_slices]
        assert len(set(nums)) == 1

    def test(self):
        self.model = self.model.eval()
        ct_slices = list()
        mask_slices = list()
        with torch.no_grad():
            for data in self._extract_slices():
                ct_mask = self.model(data)
                ct = ct_mask[:, 0:1, ...]
                mask = torch.sigmoid(ct_mask[:, 1:2, ...])
                ct_slices.append(ct.cpu().numpy())
                mask_slices.append(mask.cpu().numpy())
        ct = self.combine_slices(ct_slices)
        mask = self.combine_slices(mask_slices)
        return ct, mask

    def _extract_slices(self):
        num_slices = self.input_slices[0].shape[self._axis]
        for start_ind in range(0, num_slices, self.args.batch_size):
            stop_ind = min(start_ind + self.args.batch_size, num_slices)
            data = [list() for _ in range(len(self.input_slices))]
            for ind in range(start_ind, stop_ind):
                for i, slices in enumerate(self.input_slices):
                    data[i].append(slices.extract_slice(self._axis, ind).data)
            data = [np.stack(d) for d in data]
            data = np.stack(data, 1)
            data = torch.tensor(data).float().to(device=self._device)
            yield data


class TesterCT_(TesterBoth_):
    def test(self):
        self.model = self.model.eval()
        ct_slices = list()
        with torch.no_grad():
            for data in self._extract_slices():
                ct = self.model(data)
                ct_slices.append(ct.cpu().numpy())
        ct = self.combine_slices(ct_slices)
        return ct


class TesterMask_(TesterBoth_):
    def test(self):
        self.model = self.model.eval()
        mask_slices = list()
        ls_slices = list()
        with torch.no_grad():
            for data in self._extract_slices():
                mask, ls, edge = self.model(data)
                # mask = torch.sigmoid(mask)
                mask_slices.append(mask.cpu().numpy())
                ls_slices.append(ls.cpu().numpy())
        mask = self.combine_slices(mask_slices)
        ls = self.combine_slices(ls_slices)
        return mask, ls
