import torch
import re
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage.measurements import label as find_cc
from copy import deepcopy
from collections import OrderedDict

from .network import UNet
from .dataset import ImageSlices
from .utils import padcrop


class Tester:
    def __init__(self, args):
        self.args = args
        self._parse_args()
        self._load_model()
        self._create_testers()
        self._create_combine_images()

    def _parse_args(self):
        Path(self.args.output_dir).mkdir(exist_ok=True, parents=True)
        with open(self.args.train_config) as f:
            self._config = json.load(f)

        if self.args.use_cuda:
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

    def _load_model(self):
        in_channels = len(self._config['parsed_in_data_mode']) \
            * self._config['stack_size']
        self._model = UNet(
            in_channels,
            self._config['parsed_out_data_mode_dict'],
            self._config['num_trans_down'],
            self._config['num_channels'],
            num_out_hidden=self._config['num_out_hidden']
        ).to(self._device)
        self._cp = torch.load(self.args.checkpoint, map_location=self._device)
        self._model.load_state_dict(self._cp['model_state_dict'])

    def _create_testers(self):
        all_slices = list()
        self._nifti = list()
        target_shape = self.args.target_shape
        for fn in self.args.images:
            nii_obj = nib.load(fn)
            im_slices = ImageSlices(nii_obj, '', target_shape=target_shape)
            all_slices.append(im_slices)
            self._nifti.append(nii_obj)

        self._testers = list()
        for axis in range(3):
            self._testers.append(self._create_tester(all_slices, axis))

    def _create_tester(self, all_slices, axis):
        return Tester_(self._model, all_slices, CombineSlices(axis), self.args)

    def _create_combine_images(self):
        if self.args.combine == 'median':
            self._combine_images = CalcMedian()
        elif self.args.combine == 'mean':
            self._combine_images = CalcMean()

    def test(self):
        formatted_pred = OrderedDict()
        for axis, tester in enumerate(self._testers):
            pred = tester.test()
            pred = [padcrop(ct, self._nifti[0].shape) for p in pred]
            for name, attrs in self._config['parsed_out_data_mode_dict'].items():
                attrs.append('edge')
                if name not in formatted_pred:
                    formatted_pred[name] = OrderedDict()
                for p, attr in zip(pred, attrs):
                    if attr not in formatted_pred[name]:
                        formatted_pred[name] = list()
                    formatted_pred[name].append(p)

        for name, pred_all in formatted_pred.items():
            for attr, pred_single_image in pred_all.items():
                name = '-'.join([name, attr])
                for axis, pred_single_axis in enumerate(pred_single_image):
                    self._save_image(pred_single_axis, f'{name}_axis-{axis}')
                comb = self._combine_images(pred_single_image)
                self._save_image(comb, f'{name}')

    def _save_image(self, im, name):
        filename = Path(self.args.t1w).name
        filename = re.sub(r'\.nii(\.gz)*$', '', filename)
        filename = '_'.join([filename, name])
        filename = Path(self.args.output_dir, filename).with_suffix('.nii.gz')
        obj = nib.Nifti1Image(im, self._nifti[0].affine, self._nifti[0].header)
        obj.to_filename(filename)

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


class Tester_:
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
        with torch.no_grad():
            results = OrderedDict()
            for data in self._extract_slices():
                print(data.shape)
                pred = self.model(data)
                for name, p in pred.items():
                    if name not in results:
                        results[name] = [[] for _ in range(len(p))]
                    for i, pp in enumerate(p):
                        results[name][i].append(pp.cpu().numpy())

        for name, r in results.items():
            for i, rr in enumerate(r):
                results[name][i] = self.combine_slices(rr)
        return results

    def _extract_slices(self):
        num_slices = self.input_slices[0].shape[self._axis]
        print('num_slices', num_slices)
        for start_ind in range(0, num_slices, self.args.batch_size):
            stop_ind = min(start_ind + self.args.batch_size, num_slices)
            data = [list() for _ in range(len(self.input_slices))]
            for ind in range(start_ind, stop_ind):
                for i, slices in enumerate(self.input_slices):
                    d = slices.extract_slice(self._axis, ind).data
                    data[i].append(d)
            data = [np.stack(d) for d in data]
            data = np.stack(data, 1)
            data = torch.tensor(data).float().to(device=self._device)
            yield data
