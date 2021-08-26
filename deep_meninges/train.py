import json
import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
from collections import deque

from .network import UNet
from .dataset import create_dataset, FlipLR, Compose, Scale
from .contents import ContentsBuilder, ContentsBuilderValid


torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer:
    def __init__(self, args):
        self.args = args
        self._create_train_loader()
        self._add_more_args()
        self._save_args()
        self._create_model()
        self._create_optim()
        self._create_loss_funcs()
        self._create_contents()
        self._load_pre_checkpoint()
        self._save_arch()

    def train(self):
        self._contents.start_observers()
        for i in self._contents.counter['epoch']:
            self._train_epoch()
        self._contents.close_observers()

    def _train_epoch(self):
        for j, data in enumerate(self._train_loader):
            self._contents.counter['batch'].index0 = j

            self._model.train()
            self._optim.zero_grad()

            if self.args.output_data_mode == 'ct_mask':
                self._train_batch_both(data[:-2], data[-2], data[-1])
            elif self.args.output_data_mode == 'ct':
                self._train_batch_ct_only(data[:-1], data[-1])
            elif self.args.output_data_mode == 'mask':
                self._train_batch_mask_only(data[:-1], data[-1])

            self._contents.notify_observers()

    def _apply_network_t1w_t2w(self, input_data):
        self._t1w, self._t2w = input_data
        t1w_data = self._t1w.data.cuda()
        t2w_data = self._t2w.data.cuda()
        t1w_t2w = torch.cat((t1w_data, t2w_data), dim=1)
        pred = self._model(t1w_t2w)
        return pred

    def _apply_network_t1w(self, input_data):
        self._t1w = input_data[0]
        pred = self._model(self._t1w.data.cuda())
        return pred

    def _apply_network_t2w(self, input_data):
        self._t2w = input_data[0]
        pred = self._model(self._t2w.data.cuda())
        return pred

    def _apply_network(self, input_data):
        if self.args.input_data_mode == 't1w':
            pred = self._apply_network_t1w(input_data)
        elif self.args.input_data_mode == 't2w':
            pred = self._apply_network_t2w(input_data)
        elif self.args.input_data_mode == 't1w_t2w':
            pred = self._apply_network_t1w_t2w(input_data)
        return pred

    def _apply_network_train(self, input_data):
        pred = self._apply_network(input_data)
        if 't1w' in self.args.input_data_mode:
            self._contents.set_tensor_cpu('t_t1w', self._t1w.data, self._t1w.name)
        if 't2w' in self.args.input_data_mode:
            self._contents.set_tensor_cpu('t_t2w', self._t2w.data, self._t2w.name)
        return pred

    def _train_batch_both(self, input_data, ct, mask):
        pred = self._apply_network_train(input_data)
        ct_pred = pred[:, 0:1, ...]
        mask_pred = pred[:, 1:2, ...]

        closs, mloss, loss = self._calc_losses(ct_pred, mask_pred, ct, mask)
        loss.backward()
        self._optim.step()

        self._contents.set_value('t_ct_loss', closs.item())
        self._contents.set_value('t_mask_loss', mloss.item())
        self._contents.set_value('t_total_loss', loss.item())

        self._contents.set_tensor_cpu('t_ct', ct.data, ct.name)
        self._contents.set_tensor_cpu('t_mask', mask.data, mask.name)
        self._contents.set_tensor_cuda('t_ct_pred', ct_pred, ct.name)
        self._contents.set_tensor_cuda('t_mask_pred', mask_pred, mask.name)

    def _train_batch_mask_only(self, input_data, mask):
        mask_pred = self._apply_network_train(input_data)
        loss = self._mask_loss_func(mask_pred, mask.data.cuda())
        loss.backward()
        self._optim.step()

        self._contents.set_value('t_mask_loss', loss.item())
        self._contents.set_tensor_cpu('t_mask', mask.data, mask.name)
        self._contents.set_tensor_cuda('t_mask_pred', mask_pred, mask.name)

    def _train_batch_ct_only(self, input_data, ct):
        ct_pred = self._apply_network_train(input_data)
        loss = self._ct_loss_func(ct_pred, ct.data.cuda())
        loss.backward()
        self._optim.step()

        self._contents.set_value('t_ct_loss', loss.item())
        self._contents.set_tensor_cpu('t_ct', ct.data, ct.name)
        self._contents.set_tensor_cuda('t_ct_pred', ct_pred, ct.name)

    def _add_more_args(self):
        Path(self.args.output_dir).mkdir(exist_ok=True, parents=True)
        self.args.output_checkpoint_dir = str(Path(self.args.output_dir, 'checkpoints'))
        self.args.loss_filename = str(Path(self.args.output_dir, 'loss.csv'))
        self.args.config_filename = str(Path(self.args.output_dir, 'config.json'))
        self.args.train_image_dir = str(Path(self.args.output_dir, 'train_images'))
        self.args.valid_image_dir = str(Path(self.args.output_dir, 'valid_images'))
        self.args.arch_filename = str(Path(self.args.output_dir, 'arch.txt'))
        self.args.num_batches = len(self._train_loader)

    def _save_arch(self):
        with open(self.args.arch_filename, 'w') as txt:
            txt.write(self._model.__str__())

    def _save_args(self):
        result = dict()
        for arg in vars(self.args):
            result[arg] = getattr(self.args, arg)
        with open(self.args.config_filename, 'w') as jfile:
            json.dump(result, jfile, indent=4)

    def _create_model(self):
        num_in_chan = 2 if self.args.input_data_mode == 't1w_t2w' else 1
        num_out_chan = 2 if self.args.output_data_mode == 'ct_mask' else 1
        self._model = UNet(num_in_chan, num_out_chan, 5,
                           self.args.num_channels).cuda()
        if self.args.finetune:
            for name, param in self._model.named_parameters():
                if name.startswith('out0'):
                    param.requires_grad = False

    def _create_optim(self):
        self._optim = Adam(self._model.parameters(), lr=self.args.learning_rate,
                           betas=(0.5, 0.999))

    def _create_loss_funcs(self):
        self._ct_loss_func = torch.nn.L1Loss()
        self._mask_loss_func = torch.nn.BCEWithLogitsLoss()

    def _create_contents(self):
        builder = ContentsBuilder(self._model, self._optim, self.args)
        self._contents = builder.build().contents

    def _load_pre_checkpoint(self):
        if self.args.checkpoint and Path(self.args.checkpoint).is_file():
            cp = torch.load(self.args.checkpoint)
            self._contents.load_state_dicts(cp)

    def _create_train_loader(self):
        target_shape = self.args.target_shape
        skip = self._get_skip_im_types()
        transform = Compose([FlipLR(), Scale(self.args.scale_aug)])
        ds = create_dataset(self.args.train_dir, target_shape, transform,
                            skip=skip, memmap=self.args.memmap)
        dl = DataLoader(ds, batch_size=self.args.batch_size,
                        num_workers=self.args.num_workers, shuffle=True)
        self._train_loader = dl

    def _calc_losses(self, ct_pred, mask_pred, ct, mask):
        ct_loss = self._ct_loss_func(ct_pred, ct.data.cuda())
        mask_loss = self._mask_loss_func(mask_pred, mask.data.cuda())

        if self.args.finetune:
            loss = self.args.lambda_mask * mask_loss
        else:
            loss = self.args.lambda_ct * ct_loss + self.args.lambda_mask * mask_loss

        return ct_loss, mask_loss, loss

    def _get_skip_im_types(self):
        skip = list()
        if 't1w' not in self.args.input_data_mode:
            skip.append('t1w')
        if 't2w' not in self.args.input_data_mode:
            skip.append('t2w')
        if 'ct' not in self.args.output_data_mode:
            skip.append('ct')
        if 'mask' not in self.args.output_data_mode:
            skip.append('mask')
        return skip


class TrainerValid(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self._create_valid_loader()

    def _create_contents(self):
        builder = ContentsBuilderValid(self._model, self._optim, self.args)
        self._contents = builder.build().contents

    def _create_valid_loader(self):
        target_shape = self.args.target_shape
        skip = self._get_skip_im_types()
        ds = create_dataset(self.args.valid_dir, target_shape, skip=skip,
                            shuffle_once=True, memmap=self.args.memmap)
        dl = DataLoader(ds, batch_size=self.args.batch_size,
                        num_workers=self.args.num_workers, shuffle=True)
        self._valid_loader = dl

    def _train_epoch(self):
        for j, data in enumerate(self._train_loader):
            self._contents.counter['batch'].index0 = j

            self._model.train()
            self._optim.zero_grad()

            if self.args.output_data_mode == 'ct_mask':
                self._train_batch_both(data[:-2], data[-2], data[-1])
            elif self.args.output_data_mode == 'ct':
                self._train_batch_ct_only(data[:-1], data[-1])
            elif self.args.output_data_mode == 'mask':
                self._train_batch_mask_only(data[:-1], data[-1])

            if self._needs_to_valid():
                self._model.eval()
                self._valid_epoch()

            self._contents.notify_observers()

    def _needs_to_valid(self):
        epoch_ind = self._contents.counter['epoch'].index1
        batch_ind = self._contents.counter['batch'].index1
        rule1 = epoch_ind % self.args.valid_step == 0
        rule2 = self._contents.counter['batch'].has_reached_end()
        return rule1 and rule2

    def _apply_network_valid(self, input_data):
        pred = self._apply_network(input_data)
        if 't1w' in self.args.input_data_mode:
            self._contents.set_tensor_cpu('v_t1w', self._t1w.data, self._t1w.name)
        if 't2w' in self.args.input_data_mode:
            self._contents.set_tensor_cpu('v_t2w', self._t2w.data, self._t2w.name)
        return pred

    def _valid_epoch(self, input_data):
        if self.args.output_data_mode == 'ct_mask':
            self._valid_epoch_both(input_data)
        elif self.args.output_data_mode == 'ct':
            self._valid_epoch_ct_only(input_data)
        elif self.args.output_data_mode == 'mask':
            self._valid_epoch_mask_only(input_data)

    def _valid_epoch_both(self):
        ct_losses = list()
        mask_losses = list()
        losses = list()
        for j, data in enumerate(self._valid_loader):
            with torch.no_grad():
                ct = data[-2]
                mask = data[-1]
                input_data = data[:-2]
                pred = self._apply_network_valid(input_data)

                ct_pred = pred[:, 0:1, ...]
                mask_pred = pred[:, 1:2, ...]

            closs, mloss, loss = self._calc_losses(ct_pred, mask_pred, ct, mask)
            ct_losses.append(closs.item())
            mask_losses.append(mloss.item())
            losses.append(loss.item())

        self._contents.set_tensor_cpu('v_ct', ct.data, ct.name)
        self._contents.set_tensor_cpu('v_mask', mask.data, mask.name)
        self._contents.set_tensor_cuda('v_ct_pred', ct_pred, ct.name)
        self._contents.set_tensor_cuda('v_mask_pred', mask_pred, mask.name)

        self._contents.set_value('v_ct_loss', np.mean(ct_losses))
        self._contents.set_value('v_mask_loss', np.mean(mask_losses))
        self._contents.update_valid_loss(np.mean(losses))

    def _valid_epoch_mask_only(self):
        mask_losses = list()
        for j, data in enumerate(self._valid_loader):
            with torch.no_grad():
                mask = data[-1]
                input_data = data[:-1]
                mask_pred = self._apply_network_valid(input_data)
                loss = self._mask_loss_func(mask_pred, mask.data.cuda())
            mask_losses.append(loss.item())
        self._contents.set_tensor_cpu('v_mask', mask.data, mask.name)
        self._contents.set_tensor_cuda('v_mask_pred', mask_pred, mask.name)
        self._contents.set_value('v_mask_loss', np.mean(mask_losses))
        self._contents.update_valid_loss(np.mean(mask_losses))

    def _valid_epoch_ct_only(self):
        ct_losses = list()
        for j, data in enumerate(self._valid_loader):
            with torch.no_grad():
                ct = data[-1]
                input_data = data[:-1]
                ct_pred = self._apply_network_valid(input_data)
                loss = self._ct_loss_func(ct_pred, ct.data.cuda())
            ct_losses.append(loss.item())
        self._contents.set_tensor_cpu('v_ct', ct.data, ct.name)
        self._contents.set_tensor_cuda('v_ct_pred', ct_pred, ct.name)
        self._contents.set_value('v_ct_loss', np.mean(ct_losses))
        self._contents.update_valid_loss(np.mean(ct_losses))
