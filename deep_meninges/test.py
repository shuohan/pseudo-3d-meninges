import torch
import re
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage.measurements import label as find_cc
from copy import deepcopy
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from nighres.shape import topology_correction

from .network import UNet
from .dataset import SubjectData
from .utils import padcrop


class Tester:
    def __init__(self, args):
        self.args = args
        self._parse_args()
        self._get_device()
        self._load_model()
        self._create_tester()
        self._create_combine_images()

    def _parse_args(self):
        Path(self.args.output_dir).mkdir(exist_ok=True, parents=True)
        with open(self.args.train_config) as f:
            self._config = json.load(f)

    def _get_device(self):
        device = 'cuda' if self.args.use_cuda else 'cpu'
        self._device = torch.device(device)

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

    def _get_filenames(self):
        filenames = OrderedDict()
        for i, im_type in enumerate(self._config['parsed_in_data_mode']):
            filenames[im_type] = [self.args.images[i]]
        return filenames

    def _create_tester(self):
        subj_data = SubjectData(
            'test',
            self._get_filenames(),
            target_shape=self.args.target_shape,
            stack_size=self._config['stack_size']
        )
        loader = DataLoader(
            subj_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False
        )
        self._config['orig_shape'] = nib.load(self.args.images[0]).shape
        self._tester = Tester_(self._model, loader, self._device, self._config)

    def _create_combine_images(self):
        if self.args.combine == 'median':
            self._combine_images = CalcMedian()
        elif self.args.combine == 'mean':
            self._combine_images = CalcMean()

    def test(self):
        pred = self._tester.test()
        comb_pred = OrderedDict()
        for name, attrs in self._config['parsed_out_data_mode_dict'].items():
            attrs.append('edge')
            if name not in comb_pred:
                comb_pred[name] = list()
            for i, attr in enumerate(attrs):
                imname = '-'.join([name, attr])
                chunks = pred[name][i]
                for axis, chunk in enumerate(chunks):
                    chunk_name = '_'.join([imname, f'axis-{axis}'])
                    self._save_image(chunk, chunk_name)
                comb_chunk = self._combine_images(chunks)
                comb_pred[name].append(comb_chunk)
                self._save_image(comb_chunk, imname)

        outer_mask = 1
        prod_masks = list()
        for name, attrs in self._config['parsed_out_data_mode_dict'].items():
            mask_ind = attrs.index('mask')
            outer_mask = comb_pred[name][mask_ind] * outer_mask
            prod_masks.append(outer_mask)
            imname = '-'.join([name, attrs[mask_ind]]) + '_prod'
            self._save_image(outer_mask, imname)
        stacked_mask = np.sum(prod_masks, axis=0)
        fn = self._save_image(stacked_mask, 'stacked-mask')

        tpc = topology_correction(
            fn, 'probability_map',
            overwrite=True,
            save_data=True,
            output_dir=self.args.output_dir,
            file_name=Path(fn).name
        )


    def _save_image(self, im, name):
        obj = nib.load(self.args.images[0])
        filename = Path(self.args.images[0]).name
        filename = re.sub(r'\.nii(\.gz)*$', '', filename)
        filename = '_'.join([filename, name])
        filename = Path(self.args.output_dir, filename).with_suffix('.nii.gz')
        print('Save', filename)
        out_obj = nib.Nifti1Image(im, obj.affine, obj.header)
        out_obj.to_filename(filename)
        return str(filename)

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


class Tester_:
    def __init__(self, model, loader, device, config):
        self.model = model
        self.loader = loader
        self.device = device
        self.config = config

    def test(self):
        self.model = self.model.eval()
        predictions = OrderedDict()
        with torch.no_grad():
            for data in tqdm(self.loader):
                data = [i.data for i in data]
                data = torch.cat(data, dim=1).to(self.device)
                pred = self.model(data)
                self._add_slices(pred, predictions)
        self._chunk_predictions(predictions)
        return predictions

    def _add_slices(self, pred, predictions):
        for name, ps in pred.items():
            if name not in predictions:
                predictions[name] = [[] for _ in ps]
            for i, p in enumerate(ps):
                predictions[name][i].append(p.cpu().numpy())

    def _chunk_predictions(self, predictions):
        shape = self.loader.dataset.shape
        for name, ps in predictions.items():
            for i, p in enumerate(ps):
                p = np.concatenate(p, axis=0).squeeze(1)
                chunks = list()
                start_ind = 0
                for axis, s in enumerate(shape):
                    stop_ind = start_ind + s
                    chunk = p[start_ind : stop_ind]
                    chunk = self._transpose(chunk, axis)
                    chunks.append(chunk)
                    start_ind = stop_ind

                chunks = [padcrop(c, self.config['orig_shape'], False)
                          for c in chunks]
                predictions[name][i] = chunks

    def _transpose(self, chunk, axis):
        if axis in [0, 1]: # top bottom swap
            chunk = np.flip(chunk, -1)
        if axis == 1:
            chunk = np.transpose(chunk, (1, 0, 2))
        elif axis == 2:
            chunk = np.transpose(chunk, (1, 2, 0))
        return chunk
