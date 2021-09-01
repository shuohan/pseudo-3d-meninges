from copy import deepcopy
from ptxl.abstract import Contents
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
        self._contents_cls = Contents
        self._checkpont_saver_cls = CheckpointSaver
        self._init_tensor_attrs()
        self._init_value_attrs()

    def _init_tensor_attrs(self):
        in_attrs = self.args.input_data_mode.split('_')
        out_attrs = self._parse_out_data_mode()
        self._tensor_attrs = in_attrs + out_attrs

    def _init_value_attrs(self):
        attrs = self._parse_out_data_mode()
        self._value_attrs = list()
        for k, v in attrs.items():
            self._value_attrs.append('_'.join(['t', k, v]))

    def _get_out_attrs(self):
        attrs = self._parse_out_data_mode()
        for attr_key, attr_val in attrs.items():
            assert 'mask' in attr_val
            assert 'sdf' in attr_val
        for k in attrs.keys():
            attrs[k].append('edge')
            attrs[k].extend([v + '_pred' for v in attrs[k]])
        return attrs

    def _parse_out_data_mode(self):
        attrs_tmp = self.args.output_data_mode.split('_')
        attrs = dict()
        for attr in attrs_tmp:
            k, v = attr.split('-')
            if k not in attrs:
                attrs[k] = list()
            attrs.append(v)
        return attrs

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
        contents = self._contents_cls(
            self.model,
            self.optim,
            counter,
            self.args.input_data_mode,
            self.args.output_data_mode
        )
        return contents

    def _set_observers(self):
        self._set_printer()
        self._set_logger()
        self._set_checkpoint_saver()
        self._set_train_savers()

    def _set_printer(self):
        attrs = self._value_attrs()
        printer = MultiTqdmPrinter(attrs=attrs)
        self.contents.register(printer)

    def _set_logger(self):
        attrs = self._value_attrs()
        logger = Logger(self.args.loss_filename, attrs=attrs)
        self.contents.register(logger)

    def _get_value_attrs(self):
        return self.contents.get_value_attrs()

    def _set_checkpoint_saver(self):
        cp_saver = self._checkpont_saver_cls(
            self.args.output_checkpoint_dir,
            step=self.args.checkpoint_save_step
        )
        self.contents.register(cp_saver)

    def _set_train_savers(self):
        self._set_im_savers(
            self.args.train_image_dir,
            self.args.image_save_step, 't'
        )

    def _set_im_savers(self, dirname, step, prefix):
        save_png = create_save_image('png_norm', 'image')
        attrs = ['_'.join([prefix, a]) for a in self._tensor_attrs]
        saver = ImageSaver(dirname, save_png, attrs=attrs, step=step)
        self.contents.register(saver)


class ContentsBuilderValid(ContentsBuilder):
    def _set_observers(self):
        super()._set_observers()
        self._contents_cls = ContentsValid
        self._checkpont_saver_cls = CheckpointSaverValid

    def _set_observers(self):
        super()._set_observers()
        self._set_valid_savers()

    def _set_valid_savers(self):
        self._set_im_savers(
            self.args.valid_image_dir,
            self.args.valid_save_step, 'v'
        )


class ContentsValid(Contents):
    def __init__(self, model, optim, counter):
        super().__init__(model, optim, counter)
        self.best_model_state = self.model.state_dict()
        self.best_optim_state = self.optim.state_dict()

    def _init_value_attrs(self):
        attrs = self._parse_out_data_mode()
        self._value_attrs = list()
        for k, v in attrs.items():
            self._value_attrs.append('_'.join(['t', k, v]))
            self._value_attrs.append('_'.join(['v', k, v]))

    def update_valid_loss(self, valid_loss):
        valid_loss = valid_loss
        self.set_value('v_total_loss', valid_loss)
        if valid_loss < self.get_value('min_v_loss'):
            self.set_value('min_v_loss', valid_loss)
            epoch_ind = self.counter['epoch'].index1
            self.set_value('min_v_epoch', epoch_ind)
            self.best_model_state = deepcopy(self.model.state_dict())
            self.best_optim_state = deepcopy(self.optim.state_dict())

    def get_model_state_dict(self):
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
