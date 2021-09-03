import json
import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from pathlib import Path
from collections import OrderedDict

from .network import UNet
from .dataset import create_dataset_multi, FlipLR, Compose, Scale
from .contents import ContentsBuilder, ContentsBuilderValid


torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer:
    def __init__(self, args):
        self.args = args
        self._parse_data_modes()
        self._parse_loss_lambdas()
        self._create_train_loader()
        self._add_more_args()
        self._save_args()
        self._create_model()
        self._create_optim()
        self._create_loss_funcs()
        self._create_contents()
        self._load_pre_checkpoint()
        self._save_arch()

        self._slice_ind = self.args.stack_size // 2

    def train(self):
        self._contents.start_observers()
        for i in self._contents.counter['epoch']:
            self._train_epoch()
            self._train_loader.dataset.update()
        self._contents.close_observers()

    def _train_epoch(self):
        for j, data in enumerate(self._train_loader):
            self._contents.counter['batch'].index0 = j
            self._model.train()
            self._optim.zero_grad()

            input_data, true_data = self._split_data(data)
            pred, losses, total_loss = self._train_batch(input_data, true_data)

            self._record_input_data(input_data, 't')
            self._record_true_data(true_data, 't')
            self._record_predictions(pred, 't')
            self._record_losses(losses, total_loss, 't')
            self._contents.notify_observers()

    def _train_batch(self, input_data, true_data):
        pred = self._apply_network(input_data)
        losses = self._calc_total_loss(pred, true_data)
        total_loss = self._sum_losses(losses)
        total_loss.backward()
        self._optim.step()
        return pred, losses, total_loss

    def _split_data(self, data):
        num_input_data = len(self.args.parsed_in_data_mode)
        input_data = data[:num_input_data]

        start_ind = num_input_data
        true_data = OrderedDict()
        for outname, attrs in self.args.parsed_out_data_mode_dict.items():
            stop_ind = start_ind + len(attrs)
            # print('split data', start_ind, stop_ind, len(data))
            true_data[outname] = data[start_ind : stop_ind]
            start_ind = stop_ind

        return input_data, true_data

    def _apply_network(self, input_data, prefix='t'):
        data = [i.data for i in input_data]
        data = torch.cat(data, dim=1).cuda()
        pred = self._model(data)
        return pred

    def _sum_losses(self, losses):
        total_loss = 0
        for outname in self.args.parsed_out_data_mode_dict.keys():
            loss = losses[outname]
            lamb = self.args.loss_lambdas[outname]
            for l, w in zip(loss, lamb):
                total_loss += w * l
        return total_loss

    def _record_input_data(self, input_data, prefix='t'):
        for data, attr in zip(input_data, self.args.parsed_in_data_mode):
            attr = '_'.join([prefix, attr])
            data_d = data.data[:, self._slice_ind, ...]
            self._contents.set_tensor_cpu(attr, data_d, data.name)

    def _record_true_data(self, true_data, prefix='t'):
        for outname, attrs in self.args.parsed_out_data_mode_dict.items():
            for t, attr in zip(true_data[outname], attrs):
                attr = '-'.join([outname, attr])
                attr = '_'.join([prefix, attr])
                if len(t) == 0:
                    self._contents.set_tensor_cpu(attr, None, '')
                else:
                    data = t.data[:, self._slice_ind, ...]
                    self._contents.set_tensor_cpu(attr, data, t.name)

    def _record_predictions(self, pred, prefix='t'):
        for outname, attrs in self.args.parsed_out_data_mode_dict.items():
            for p, attr in zip(pred[outname], attrs + ['edge']):
                attr = '-'.join([outname, attr])
                attr = '_'.join([prefix, attr, 'pred'])
                with torch.no_grad():
                    # print(attr, p.shape)
                    self._contents.set_tensor_cuda(attr, p.cpu(), '')

    def _record_losses(self, losses, total_loss, prefix='t'):
        for outname, attrs in self.args.parsed_out_data_mode_dict.items():
            for loss, attr in zip(losses[outname], attrs):
                attr = '-'.join([outname, attr])
                attr = '_'.join([prefix, attr])
                if isinstance(loss, torch.Tensor):
                    loss_val = loss.item()
                elif loss == 0:
                    loss_val = float('nan')
                self._contents.set_value(attr, loss_val)
        attr = '_'.join([prefix, 'total_loss'])
        self._contents.set_value(attr, total_loss.item())

    def _calc_total_loss(self, pred, true_data):
        losses = OrderedDict()
        for outname, attrs in self.args.parsed_out_data_mode_dict.items():
            pred_tmp = pred[outname][:-1]
            pred_edge = pred[outname][-1]
            truth_tmp = true_data[outname]
            if outname not in losses:
                losses[outname] = list()
            for p, t, a in zip(pred_tmp, truth_tmp, attrs):
                if len(t) == 0:
                    loss = 0
                else:
                    t_data = t.data[:, self._slice_ind : self._slice_ind+1, ...]
                    loss = self._calc_loss(p, t_data.cuda(), a, pred_edge)
                losses[outname].append(loss)
        return losses

    def _calc_loss(self, pred, truth, attr, pred_edge):
        if attr == 'mask':
            return self._mask_loss_func(pred, truth)
        elif attr == 'sdf':
            return self._calc_sdf_loss(pred, truth, pred_edge)
        else:
            return self._default_loss_func(pred, truth)

    def _calc_sdf_loss(self, pred_sdf, true_sdf, pred_edge):
        error = pred_edge * (pred_sdf - true_sdf)
        sdf_loss = torch.mean(torch.abs(error))
        return sdf_loss

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
        in_channels = len(self.args.parsed_in_data_mode) * self.args.stack_size
        self._model = UNet(
            in_channels,
            self.args.parsed_out_data_mode_dict,
            self.args.num_trans_down,
            self.args.num_channels,
            num_out_hidden=self.args.num_out_hidden
        ).cuda()
        if self.args.finetune:
            for name, param in self._model.named_parameters():
                if name.startswith('out0'):
                    param.requires_grad = False

    def _create_optim(self):
        self._optim = Adam(self._model.parameters(), lr=self.args.learning_rate,
                           betas=(0.5, 0.999))

    def _create_loss_funcs(self):
        self._mask_loss_func = torch.nn.BCELoss()
        self._default_loss_func = torch.nn.L1Loss()

    def _create_contents(self):
        builder = ContentsBuilder(self._model, self._optim, self.args)
        self._contents = builder.build().contents

    def _load_pre_checkpoint(self):
        if self.args.checkpoint and Path(self.args.checkpoint).is_file():
            cp = torch.load(self.args.checkpoint)
            self._contents.load_state_dicts(cp)

    def _parse_data_modes(self):
        self.args.parsed_in_data_mode = self.args.input_data_mode.split('_')
        self.args.parsed_out_data_mode = self.args.output_data_mode.split('_')
        self.args.parsed_out_data_mode_dict = OrderedDict()
        for attr in self.args.parsed_out_data_mode:
            k, v = attr.split('-')
            if k not in self.args.parsed_out_data_mode_dict:
                self.args.parsed_out_data_mode_dict[k] = list()
            self.args.parsed_out_data_mode_dict[k].append(v)
        for attr_key, attr_val in self.args.parsed_out_data_mode_dict.items():
            assert 'mask' in attr_val
            assert 'sdf' in attr_val

    def _parse_loss_lambdas(self):
        lambdas = OrderedDict()
        outnames = self.args.parsed_out_data_mode_dict.keys()
        for outname, lamb in zip(outnames, self.args.loss_lambdas):
            lambdas[outname] = list(map(float, lamb.split(',')))
        self.args.loss_lambdas = lambdas

    def _define_loading_order(self):
        return self.args.parsed_in_data_mode + self.args.parsed_out_data_mode

    def _create_train_loader(self):
        target_shape = self.args.target_shape
        transform = Compose([FlipLR(), Scale(self.args.scale_aug)])
        dataset = create_dataset_multi(
            self.args.train_dir,
            self.args.batch_size,
            self.args.num_slices_per_epoch,
            self.args.num_epochs,
            target_shape=target_shape,
            transform=transform,
            memmap=self.args.memmap,
            stack_size=self.args.stack_size,
            loading_order=self._define_loading_order(),
        )
        self._train_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            drop_last=False,
            num_workers=self.args.num_workers,
            shuffle=False # handled by dataset
        )


class TrainerValid(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self._create_valid_loader()

    def _create_contents(self):
        builder = ContentsBuilderValid(self._model, self._optim, self.args)
        self._contents = builder.build().contents

    def _create_valid_loader(self):
        target_shape = self.args.target_shape
        dataset = create_dataset(
            self.args.valid_dir,
            target_shape,
            shuffle_once=True,
            memmap=self.args.memmap,
            loading_order=self._define_loading_order(),
        )
        self._valid_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True
        )

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
                self._train_batch_mask_only(data[:-2], data[-2], data[-1])

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
        pred = self._apply_network(input_data, 'v')
        if 't1w' in self.args.input_data_mode:
            self._contents.set_tensor_cpu('v_t1w', self._t1w.data, self._t1w.name)
        if 't2w' in self.args.input_data_mode:
            self._contents.set_tensor_cpu('v_t2w', self._t2w.data, self._t2w.name)
        return pred

    def _valid_epoch(self):
        if self.args.output_data_mode == 'ct_mask':
            self._valid_epoch_both()
        elif self.args.output_data_mode == 'ct':
            self._valid_epoch_ct_only()
        elif self.args.output_data_mode == 'mask':
            self._valid_epoch_mask_only()

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
        ls_losses = list()
        losses = list()
        for j, data in enumerate(self._valid_loader):
            with torch.no_grad():
                mask = data[-2]
                ls = data[-1]
                input_data = data[:-2]
                mask_pred, ls_pred, edge = self._apply_network_valid(input_data)
                mask_loss, ls_loss, loss = self._calc_mask_losses(
                    mask_pred, ls_pred, edge, mask.data.cuda(), ls.data.cuda())
            losses.append(loss.item())
            mask_losses.append(mask_loss.item())
            ls_losses.append(ls_loss.item())

        self._contents.set_value('v_mask_loss', np.mean(mask_losses))
        self._contents.set_value('v_ls_loss', np.mean(ls_losses))
        self._contents.set_value('v_total_loss', np.mean(losses))

        self._contents.set_tensor_cpu('v_mask', mask.data, mask.name)
        self._contents.set_tensor_cuda('v_mask_pred', mask_pred, mask.name)
        self._contents.set_tensor_cpu('v_ls', ls.data, ls.name)
        self._contents.set_tensor_cuda('v_ls_pred', ls_pred, ls.name)
        self._contents.set_tensor_cuda('v_edge', edge, mask.name)

        self._contents.update_valid_loss(np.mean(losses))

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
