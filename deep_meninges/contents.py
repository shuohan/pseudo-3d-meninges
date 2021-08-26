from copy import deepcopy
from ptxl.abstract import Contents as _Contents
from ptxl.utils import Counter, Counters
from ptxl.log import Logger, MultiTqdmPrinter
from ptxl.save import ImageSaver as _ImageSaver
from ptxl.save import create_save_image
from ptxl.save import CheckpointSaver


class ContentsBuilder:
    def __init__(self, model, optim, args):
        self.model = model
        self.optim = optim
        self.args = args

        zoom = {'zoom': args.image_save_zoom}
        self._save_png = create_save_image('png_norm', 'image', zoom)
        self._save_seg = create_save_image('png', 'sigmoid', zoom)
        # self._save_seg = create_save_image('png_norm', 'image', zoom)

    @property
    def contents(self):
        return self._contents

    def build(self):
        counter = self._create_counter()
        self._contents = self._create_contents(counter)
        self._set_observers()
        return self

    def _create_counter(self):
        epoch_counter = Counter('epoch', self.args.num_epochs)
        batch_counter = Counter('batch', self.args.num_batches)
        counter = Counters([epoch_counter, batch_counter])
        return counter

    def _create_contents(self, counter):
        return Contents(self.model, self.optim, counter,
                        self.args.input_data_mode, self.args.output_data_mode)

    def _set_observers(self):
        self._set_printer()
        self._set_logger()
        self._set_checkpoint_saver()
        self._set_train_savers()

    def _set_printer(self):
        attrs = self._get_value_attrs()
        printer = MultiTqdmPrinter(attrs=attrs)
        self.contents.register(printer)

    def _set_logger(self):
        attrs = self._get_value_attrs()
        logger = Logger(self.args.loss_filename, attrs=attrs)
        self.contents.register(logger)

    def _get_value_attrs(self):
        return self.contents.get_value_attrs()

    def _set_checkpoint_saver(self):
        cp_saver = CheckpointSaver(self.args.output_checkpoint_dir,
                                   step=self.args.checkpoint_save_step)
        self.contents.register(cp_saver)

    def _set_train_savers(self):
        self._set_im_savers(self.args.train_image_dir,
                            self.args.image_save_step, 't')

    def _set_im_savers(self, dirname, step, prefix):
        # im_attrs = [a for a in self.contents.attrs if 'pred' not in a]
        # seg_attrs = ['mask_pred']
        # im_attrs = ['_'.join([prefix, a]) for a in im_attrs]
        # seg_attrs = ['_'.join([prefix, a]) for a in seg_attrs]
        # print(seg_attrs, im_attrs)

        im_attrs = ['_'.join([prefix, a]) for a in self.contents.attrs]
        print(im_attrs)
        im_saver = ImageSaver(dirname, self._save_png, attrs=im_attrs, step=step)
        self.contents.register(im_saver)
        # seg_saver = ImageSaver(dirname, self._save_seg, attrs=seg_attrs,
        #                        step=step, ind_offset=len(im_attrs))
        # self.contents.register(seg_saver)


class ContentsBuilderValid(ContentsBuilder):
    def _set_observers(self):
        super()._set_observers()
        self._set_valid_savers()

    def _set_valid_savers(self):
        self._set_im_savers(self.args.valid_image_dir,
                            self.args.valid_save_step, 'v')

    def _create_contents(self, counter):
        return ContentsValid(self.model, self.optim, counter,
                             self.args.input_data_mode,
                             self.args.output_data_mode)

    def _set_checkpoint_saver(self):
        cp_saver = CheckpointSaverValid(self.args.output_checkpoint_dir,
                                        step=self.args.checkpoint_save_step)
        self.contents.register(cp_saver)


class Contents(_Contents):
    def __init__(self, model, optim, counter, input_data_mode, output_data_mode):
        super().__init__(model, optim, counter)
        self.attrs = input_data_mode.split('_')
        output_attrs = output_data_mode.split('_')
        output_pred_attrs = [oa + '_pred' for oa in output_attrs]
        self.attrs.extend(output_attrs)
        self.attrs.extend(output_pred_attrs)
        self.attrs.extend(['edge', 'ls', 'ls_pred'])

        self._cpu_attrs = [a for a in self.attrs if 'pred' not in a and a != 'edge']
        self._cuda_attrs = [a for a in self.attrs if 'pred' in a or a == 'edge']
        for attr in self._cpu_attrs:
            self.set_tensor_cpu('t_' + attr, None, name=None)
        for attr in self._cuda_attrs:
            self.set_tensor_cuda('t_' + attr, None, name=None)
        self.set_value('t_ct_loss', float('nan'))
        self.set_value('t_mask_loss', float('nan'))
        self.set_value('t_ls_loss', float('nan'))
        self.set_value('t_total_loss', float('nan'))


class ContentsValid(Contents):
    def __init__(self, model, optim, counter, input_data_mode, output_data_mode):
        super().__init__(model, optim, counter, input_data_mode, output_data_mode)
        for attr in self._cpu_attrs:
            self.set_tensor_cpu('v_' + attr, None, name=None)
        for attr in self._cuda_attrs:
            self.set_tensor_cuda('v_' + attr, None, name=None)
        self.set_value('v_ct_loss', float('nan'))
        self.set_value('v_mask_loss', float('nan'))
        self.set_value('v_ls_loss', float('nan'))
        self.set_value('v_total_loss', float('nan'))
        self.set_value('min_v_loss', float('inf'))
        self.set_value('min_v_epoch', float('nan'))
        self.best_model_state = self.model.state_dict()
        self.best_optim_state = self.optim.state_dict()

    def update_valid_loss(self, valid_loss):
        valid_loss = valid_loss
        self.set_value('v_total_loss', valid_loss)
        if valid_loss < self.get_value('min_v_loss'):
            self.set_value('min_v_loss', valid_loss)
            epoch_ind = self.counter['epoch'].index1
            self.set_value('min_v_epoch', epoch_ind)
            self.best_model_state = deepcopy(self.model.state_dict())
            self.best_optim_state = deepcopy(self.optim.state_dict())
            # print('\nkeep best', self.best_model_state['cb0.conv0.conv.weight'].sum(),
            #       self.model.state_dict()['cb0.conv0.conv.weight'].sum())

    def get_model_state_dict(self):
        # print('\nSave best into disk',
        #       self.best_model_state['cb0.conv0.conv.weight'].sum(),
        #       self.model.state_dict()['cb0.conv0.conv.weight'].sum())
        return self.best_model_state

    def get_optim_state_dict(self):
        return self.best_optim_state

    def load_state_dicts(self, checkpoint):
        super().load_state_dicts(checkpoint)
        self.best_model_state = checkpoint['model_state_dict']
        self.best_optim_state = checkpoint['optim_state_dict']


class CheckpointSaverValid(CheckpointSaver):
    def _get_contents_to_save(self):
        contents = {
            self._get_counter_name(): self._get_counter_index(),
            'min_valid_epoch': self.contents.get_value('min_v_epoch'),
            'model_state_dict': self.contents.get_model_state_dict(),
            'optim_state_dict': self.contents.get_optim_state_dict(),
            **self.kwargs
        }
        return contents


class ImageSaver(_ImageSaver):
    def _get_counter_named_index(self):
        return (self.contents.counter['epoch'].named_index1, )
