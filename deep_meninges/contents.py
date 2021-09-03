from copy import deepcopy
from ptxl.abstract import Contents, _Contents
from ptxl.utils import Counter, Counters
from ptxl.log import Logger, MultiTqdmPrinter, TqdmPrinterNoDesc
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
        attrs = deepcopy(self.args.parsed_in_data_mode)
        for key, values in self.args.parsed_out_data_mode_dict.items():
            attrs.extend(['-'.join([key, v]) for v in values])
            attrs.extend(['-'.join([key, v]) + '_pred' for v in values])
            attrs.append('-'.join([key, 'edge_pred']))
        self._tensor_attrs = attrs

    def _init_value_attrs(self):
        self._value_attrs = [
            '_'.join(['t', mode]) for mode in self.args.parsed_out_data_mode
        ]
        self._value_attrs.append('t_total_loss')

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
        return self._contents_cls( self.model, self.optim, counter)

    def _set_observers(self):
        self._set_printer()
        self._set_logger()
        self._set_checkpoint_saver()
        self._set_train_savers()

    def _set_printer(self):
        printer = MultiTqdmPrinter(attrs=self._value_attrs, decimals=2)
        self.contents.register(printer)

    def _set_logger(self):
        logger = Logger(self.args.loss_filename, attrs=self._value_attrs)
        self.contents.register(logger)

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
        save_png = create_save_image('png_norm', 'image', dict())
        attrs = ['_'.join([prefix, a]) for a in self._tensor_attrs]
        saver = ImageSaver(dirname, save_png, attrs=attrs, step=step)
        self.contents.register(saver)


class ContentsBuilderValid(ContentsBuilder):
    def __init__(self, model, optim, args):
        super().__init__(model, optim, args)
        self._contents_cls = ContentsValid
        self._checkpont_saver_cls = CheckpointSaverValid

    def build(self):
        counter = self._create_counter()
        self._contents = self._create_contents(counter)
        valid_counter = Counter('valid', self.args.num_valid_batches)
        valid_contents = ContentsValidProg(valid_counter)
        self._contents.valid_contents = valid_contents
        self._set_observers()
        return self

    def _set_observers(self):
        super()._set_observers()
        self._set_valid_savers()
        self._set_valid_printer()

    def _set_valid_savers(self):
        self._set_im_savers(
            self.args.valid_image_dir,
            self.args.valid_save_step, 'v'
        )

    def _set_valid_printer(self):
        printer = TqdmPrinterNoDesc(
            attrs=self._value_attrs, decimals=2, loc_offset=3)
        self.contents.valid_contents.register(printer)

    def _init_value_attrs(self):
        super()._init_value_attrs()
        for k in self.args.parsed_out_data_mode:
            self._value_attrs.append('_'.join(['v', k]))
        self._value_attrs.append('v_total_loss')
        self._value_attrs.append('min_v_loss')
        self._value_attrs.append('min_v_epoch')


class ContentsValid(Contents):
    def __init__(self, model, optim, counter):
        super().__init__(model, optim, counter)
        self.best_model_state = self.model.state_dict()
        self.best_optim_state = self.optim.state_dict()
        self.valid_contents = None

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

    def start_observers(self):
        super().start_observers()
        self.valid_contents.start_observers()

    def close_observers(self):
        super().close_observers()
        self.valid_contents.close_observers()


class ContentsValidProg(_Contents):
    def __init__(self, counter):
        super().__init__()
        self.counter = counter


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
