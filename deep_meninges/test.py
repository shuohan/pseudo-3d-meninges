import os
import torch
import re
import json
import numpy as np
import nibabel as nib
import threading
import types
from pathlib import Path
from scipy.ndimage.measurements import label as find_cc
from copy import deepcopy
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from queue import Queue

from .network import UNet
from .dataset import SubjectData, GroupImages
from .utils import padcrop


EPS = 1e-16


class ImageThread(threading.Thread):
    def __init__(self, queue, desc_bar):
        super().__init__()
        self.queue = queue
        self.desc_bar = desc_bar
    def run(self):
        while True:
            data = self.queue.get()
            self.queue.task_done()
            if data is None:
                break
            filename, nifti = data
            self.desc_bar.update_message(filename)
            nifti.to_filename(filename)


class Tester:
    def __init__(self, args):
        self.args = args
        self._parse_args()
        self._get_device()
        self._load_model()
        self._create_testers()
        self._create_combine_images()
        self._start_thread()

    def _start_thread(self):
        self._queue = Queue()
        self._desc_bar = tqdm(bar_format='{desc}', position=0)

        def _update_message(self, filename):
            desc = f'{self.tpc_message}saved {filename}'
            self.set_description(desc)
            self.refresh()
        def _set_tpc_message(self, message):
            self.lock.acquire()
            self.tpc_message = message
            self.lock.release()

        self._desc_bar.lock = threading.Lock()
        self._desc_bar.tpc_message = ''
        self._desc_bar.update_message = types.MethodType(
            _update_message, self._desc_bar)
        self._desc_bar.set_tpc_message = types.MethodType(
            _set_tpc_message, self._desc_bar)

        self._image_thread = ImageThread(self._queue, self._desc_bar)
        self._image_thread.start()

    def _close_thread(self):
        self._queue.put(None)
        self._image_thread.join()

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

    def _create_testers(self):
        filenames = self._get_filenames()
        self._tester = self._create_tester(filenames)

    def _create_tester(self, filenames):
        im_type_key = list(filenames.keys())[0]
        subj_data = SubjectData(
            str(filenames[im_type_key][0]),
            filenames,
            target_shape=self.args.target_shape,
            stack_size=self._config['stack_size']
        )
        loader = DataLoader(
            subj_data,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False
        )
        orig_shape = nib.load(loader.dataset.name).shape
        tester = Tester_(self._model, loader, self._device, orig_shape)
        return tester

    def _create_combine_images(self):
        if self.args.combine == 'median':
            self._combine_images = CalcMedian()
        elif self.args.combine == 'mean':
            self._combine_images = CalcMean()

    def test(self):
        self._filenames = self.args.images
        pred = self._tester.test()
        self._post_process(pred)
        self._close_thread()

    def _post_process(self, pred):
        comb_pred = self._comb_pred(pred)
        self._desc_bar.set_description()
        self._desc_bar.refresh()
        tpc = self._correct_masks_topology(comb_pred)
        self._fuse_mask_sdfs(tpc, comb_pred)

    def _comb_pred(self, pred):
        comb_pred = OrderedDict()
        for name, attrs in self._config['parsed_out_data_mode_dict'].items():
            attrs = attrs + ['edge']
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
        return comb_pred

    def _correct_masks_topology(self, comb_pred):
        outer_mask = 1
        prod_masks = list()
        for name, attrs in self._config['parsed_out_data_mode_dict'].items():
            mask_ind = attrs.index('mask')
            outer_mask = comb_pred[name][mask_ind] * outer_mask
            prod_masks.append(outer_mask)
            imname = '-'.join([name, attrs[mask_ind]]) + '_prod'
            self._save_image(outer_mask, imname)
        stacked_mask = np.sum(prod_masks, axis=0)
        fn = self._save_image(stacked_mask, 'stacked-mask', queue=False)
        tpc = self._tpc(fn)
        return tpc

    def _tpc(self, fn):
        self._desc_bar.set_tpc_message('correcting topology... ')
        output = self._join_output_filename('stacked-mask_tpc')
        os.system(f'tpc.py -i {fn} -o {output} > /dev/null')
        tpc = nib.load(output).get_fdata()
        self._desc_bar.set_tpc_message('')
        return tpc

    def _fuse_mask_sdfs(self, tpc, comb_pred):
        outer_sdf = None
        out_data_mode = self._config['parsed_out_data_mode_dict']
        for i, (name, attrs) in enumerate(out_data_mode.items()):
            tpc_mask = tpc > (0.5 + i)
            imname = f'stacked-mask_tpc_{name}-mask'
            self._save_image(tpc_mask, imname)
            sdf_ind = attrs.index('sdf')
            inner_sdf = comb_pred[name][sdf_ind]
            outer_sdf = self._intersect_sdfs(outer_sdf, inner_sdf)
            fused_sdf = self._fuse_mask_sdf(tpc_mask, outer_sdf)
            imname = f'stacked-mask_tpc_{name}-sdf'
            self._save_image(fused_sdf, imname)

    def _fuse_mask_sdf(self, mask, sdf):
        inv_mask = np.logical_not(mask)
        result = deepcopy(sdf)
        result[mask] = np.clip(sdf[mask], -self.args.max_sdf_value, -EPS)
        result[inv_mask] = np.clip(sdf[inv_mask], EPS, self.args.max_sdf_value)
        return result

    def _intersect_sdfs(self, outer_sdf, inner_sdf):
        if outer_sdf is not None:
            inner_sdf = np.maximum(outer_sdf + EPS, inner_sdf)
        return inner_sdf

    def _save_image(self, im, name, nii=False, queue=True):
        obj = nib.load(self._tester.loader.dataset.name)
        out_obj = im if nii else nib.Nifti1Image(im, obj.affine, obj.header)
        filename = self._join_output_filename(name)
        if queue:
            self._queue.put((filename, out_obj))
        else:
            self._desc_bar.update_message(filename)
            out_obj.to_filename(filename)
        return str(filename)

    def _join_output_filename(self, name):
        fn = Path(self._tester.loader.dataset.name).name
        fn = re.sub(r'\.nii(\.gz)*$', '', fn)
        fn = '_'.join([fn, name])
        fn = Path(self.args.output_dir, fn).with_suffix('.nii.gz')
        return fn


class TesterDataset(Tester):
    def _create_testers(self):
        self._testers = list()
        for filenames in self._find_images().values():
            tester = self._create_tester(filenames)
            self._testers.append(tester)

    def test(self):
        for self._tester in self._testers:
            pred = self._tester.test()
            self._post_process(pred)
        self._close_thread()

    def _find_images(self):
        loading_order = self._config['parsed_in_data_mode']
        group_images = GroupImages(self.args.dirname, loading_order)
        filenames = group_images.group()
        return filenames


class CalcMedian:
    def __call__(self, images):
        return np.median(np.stack(images, axis=0), axis=0)


class CalcMean:
    def __call__(self, images):
        return np.mean(np.stack(images, axis=0), axis=0)


class Tester_:
    def __init__(self, model, loader, device, orig_shape):
        self.model = model
        self.loader = loader
        self.device = device
        self.orig_shape = orig_shape

    def test(self):
        self.model = self.model.eval()
        predictions = OrderedDict()
        with torch.no_grad():
            desc = Path(self.loader.dataset.name).name
            for data in tqdm(self.loader, desc=desc, position=0):
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
                chunks = [padcrop(c, self.orig_shape, False) for c in chunks]
                predictions[name][i] = chunks

    def _transpose(self, chunk, axis):
        if axis in [0, 1]: # top bottom swap
            chunk = np.flip(chunk, -1)
        if axis == 1:
            chunk = np.transpose(chunk, (1, 0, 2))
        elif axis == 2:
            chunk = np.transpose(chunk, (1, 2, 0))
        return chunk
