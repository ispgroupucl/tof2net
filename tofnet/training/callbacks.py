import warnings
import numpy as np
from tofnet.data.generators import save_colored_masks
from tofnet.data.generator_maker import NamedSubset
import os
from collections import defaultdict, deque, OrderedDict
from collections.abc import Iterable
import time
import csv
import torch
import copy
from pathlib import Path
from tqdm.autonotebook import tqdm
from IPython.display import DisplayHandle
_TRAIN = 'train'
_TEST = 'test'
_PREDICT = 'predict'
import pandas as pd
from tofnet import train
import logging


class Callback(object):
    """ Abstract base class used to build new callbacks.

    Attributes:
        params (dict): Training parameters (eg. verbosity, batch size, number of epochs...).
        model (torch.nn.Module): Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.
    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        * on_epoch_end: logs include `acc` and `loss`, and
          optionally include `val_loss`
          (if validation is enabled in `fit`), and `val_acc`
          (if validation and accuracy monitoring are enabled).
        * on_batch_begin: logs include `size`,
          the number of samples in the current batch.
        * on_batch_end: logs include `loss`, and optionally `acc`
          (if accuracy monitoring is enabled).
    """

    def __init__(self):
        self.validation_data = None
        self.model = None
        self.learner = None
        self.logging = logging.getLogger(__name__)

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def set_learner(self, learner):
        self.learner = learner

    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.
        Subclasses should override for any actions to run. This function should only
        be called during train mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.
        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        # For backwards compatibility
        self.on_batch_begin(batch, logs=logs)

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.
        Subclasses should override for any actions to run.

         Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        # For backwards compatibility
        self.on_batch_end(batch, logs=logs)

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.
        Also called at the beginning of a validation batch in the `fit` methods,
        if validation data is provided.
        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.
        Also called at the end of a validation batch in the `fit` methods,
        if validation data is provided.
        Subclasses should override for any actions to run.

         Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.
        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.
        Subclasses should override for any actions to run.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.
        Subclasses should override for any actions to run.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.
        Subclasses should override for any actions to run.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.
        Subclasses should override for any actions to run.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.
        Subclasses should override for any actions to run.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.
        Subclasses should override for any actions to run.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

    def on_predict_end(self, logs=None):
        """Called at the end of prediction.
        Subclasses should override for any actions to run.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """

class ProgressCallback(Callback):
    def __init__(self, epochs=None, train_steps=None, val_steps=None):
        self.epoch_progress = tqdm(total=epochs, desc="Epoch", unit=" epochs", file=train.orig_stdout, dynamic_ncols=True)
        self.train_progress = tqdm(total=train_steps, desc="Train", unit=" batch", file=train.orig_stdout, dynamic_ncols=True)
        self.val_progress = tqdm(total=val_steps, desc="Validation", unit=" batch", file=train.orig_stdout, dynamic_ncols=True)
        self.logs = None
        self.log_display = DisplayHandle("logs")

    def refresh(self):
        self.epoch_progress.refresh()
        self.train_progress.refresh()
        self.val_progress.refresh()

    def ncols(self):
        return self.train_progress.dynamic_ncols(self.train_progress.fp)

    def on_train_begin(self, logs=None):
        self.epoch_progress.n = 0
        self.epoch_progress.total = self.params["epochs"]
        self.train_progress.total = self.params["steps"]
        self.val_progress.total = len(self.learner.val_loader)
        self.refresh()

    def on_train_end(self, logs=None):
        self.epoch_progress.close()
        self.train_progress.close()
        self.val_progress.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_progress.update()
        self.epoch_progress.set_description("Epoch {}".format(epoch+1))

    def on_epoch_end(self, epoch, logs=None):
        # logs = self.learner.metrics.summary()
        if self.logs is None:
            self.logs = pd.DataFrame(logs, index=[0])
        else:
            self.logs = self.logs.append(logs, ignore_index=True)

        self.log_display.update(self.logs.tail(1))
        # self.epoch_progress.set_postfix(logs)
        self.train_progress.n = 0
        self.val_progress.n = 0
        self.refresh()

    def on_train_batch_end(self, step, logs=None):
        logs = self.learner.metrics.summary()
        self.train_progress.update()
        self.train_progress.set_postfix(logs)

    def on_test_begin(self, logs=None):
        val_steps = len(self.learner.test_loader)
        self.val_progress = tqdm(total=val_steps, desc="Test", unit=" batch", file=train.orig_stdout, dynamic_ncols=True)

    def on_test_end(self, logs=None):
        if logs is not None:
            logs = pd.DataFrame([logs])
            self.log_display.update(logs)
        self.val_progress.close()

    def on_test_batch_end(self, step, logs=None):
        logs = self.learner.metrics.summary()
        self.val_progress.update()
        self.val_progress.set_postfix(logs)


class KapImageCheckpoint(Callback):
    """
        Callback to create image masks at every epoch
    """
    def __init__(self, dirpath, train_loader=None, val_loader=None, test_loader=None,  n_classes=None, period=10):
        super(KapImageCheckpoint, self).__init__()
        self.dirpath = dirpath / "images"
        self.period = period
        self.epochs_since_last_save = 0
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        if n_classes is None:
            raise ValueError()
        self.n_classes = n_classes
        self.target_size = test_loader.dataset.shape

    def create_masks(self, generator, save_path):
        prev_ind, new_ind = 0, 0
        for images, _ in generator:
            prev_ind, new_ind = new_ind, new_ind+images[list(images.keys())[0]].size()[0]
            fnames =  NamedSubset(generator.dataset, range(prev_ind, new_ind)) # [prev_ind:new_ind]
            self.model.eval()
            output = self.model(images)
            for idx, out_type in [(i, x) for i, x in enumerate(self.model.output_format) if x.name in {"mask", "keypoints"}]:
                name = out_type.name
                output_slice = output[idx].detach().cpu().numpy().transpose(0,2,3,1)
                save_colored_masks(
                    None, save_path, output_slice, self.target_size,
                    n_classes=self.n_classes, generator=fnames, name=out_type.full_name,
                    regr=(name=="keypoints"))
    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.val_loader is not None:
                save_path = os.path.join(self.dirpath, "val_"+str(epoch+1))
                os.makedirs(save_path)
                self.create_masks(self.val_loader, save_path)
            if self.train_loader is not None:
                save_path = os.path.join(self.dirpath, "train_"+str(epoch+1))
                os.makedirs(save_path)
                self.create_masks(self.train_loader, save_path)

    def on_test_end(self, logs=None):
        if self.test_loader is not None:
            save_path = os.path.join(self.dirpath, "test")
            os.makedirs(save_path)
            self.create_masks(self.test_loader, save_path)


class KapModelCheckpoint(Callback):
    """ Handles the saving of the model

    Arguments:
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss',
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(KapModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or 'f1' in self.monitor or \
                    'IoU' in self.monitor or 'IloU' in self.monitor or \
                        self.monitor.startswith('fmeasure') or 'iou' in self.monitor\
                    or 'recall' in self.monitor or 'precision' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.save_best_only:
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                if self.params['verbose'] > 0:
                   print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                            % (epoch + 1, self.monitor, self.best,
                                current))
                self.best = current
                self.best_weights = copy.deepcopy(self.model.state_dict())
            else:
                if self.params['verbose'] > 0:
                    print('\nEpoch %05d: %s did not improve from %0.5f' %
                            (epoch + 1, self.monitor, self.best))
        else:
            self.best_weights = copy.deepcopy(self.model.state_dict())
        if self.epochs_since_last_save >= self.period and \
            self.best_weights is not None:

            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_weights_only:
                torch.save(self.best_weights, filepath)
            else:
                raise NotImplementedError("Can only save weights")

    def on_test_begin(self, logs=None):
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

    def on_train_end(self, logs=None):
        if self.best_weights is not None and self.epochs_since_last_save > 0:
            filepath = self.filepath
            if self.save_weights_only:
                torch.save(self.best_weights, filepath)
            else:
                raise NotImplementedError("Can only save weights")

class CSVLogger(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    Example:
        ```python
        csv_logger = CSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```

    Arguments:
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = ''
        self._open_args = {'newline': '\n'}
        super(CSVLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = open(self.filename,
                                mode + self.file_flags,
                                **self._open_args)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        # if self.model.stop_training:
        #     # We set NA so that csv parsers do not fail for this last epoch.
        logs = dict([(k, logs[k] if k in logs else 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep
            fieldnames = ['epoch'] + self.keys
            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=fieldnames,
                                         dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

    def on_test_end(self, logs=None):
        filename = Path(self.filename).parent / "result.csv"
        if logs is not None:
            df = pd.DataFrame([logs])
            with open(filename, 'w') as csv_file:
                csv_file.write(df.to_csv(index=False))

class CallbackList(object):
    """Container abstracting a list of callbacks.

    Arguments:
        callbacks: List of `Callback` instances.
        queue_length: Queue length for keeping
            running statistics over callback execution time.
    """

    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length
        self.params = {}
        self.model = None
        self._reset_batch_timing()

    def _reset_batch_timing(self):
        self._delta_t_batch = 0.
        self._delta_ts = defaultdict(lambda: deque([], maxlen=self.queue_length))

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def set_learner(self, learner):
        self.learner = learner
        for callback in self.callbacks:
            callback.set_learner(learner)

    def _call_batch_hook(self, mode, hook, batch, logs=None):
        """Helper function for all batch_{begin | end} methods."""
        if not self.callbacks:
            return
        hook_name = 'on_{mode}_batch_{hook}'.format(mode=mode, hook=hook)
        if hook == 'end':
            if not hasattr(self, '_t_enter_batch'):
                self._t_enter_batch = time.time()
            # Batch is ending, calculate batch time
            self._delta_t_batch = time.time() - self._t_enter_batch

        logs = logs or {}
        t_before_callbacks = time.time()
        for callback in self.callbacks:
            batch_hook = getattr(callback, hook_name)
            batch_hook(batch, logs)
        self._delta_ts[hook_name].append(time.time() - t_before_callbacks)

        delta_t_median = np.median(self._delta_ts[hook_name])
        if (self._delta_t_batch > 0. and
           delta_t_median > 0.95 * self._delta_t_batch and
           delta_t_median > 0.1):
            warnings.warn(
                'Method (%s) is slow compared '
                'to the batch update (%f). Check your callbacks.'
                % (hook_name, delta_t_median), RuntimeWarning)

        if hook == 'begin':
            self._t_enter_batch = time.time()

    def _call_begin_hook(self, mode):
        """Helper function for on_{train|test|predict}_begin methods."""
        if mode == _TRAIN:
            self.on_train_begin()
        elif mode == _TEST:
            self.on_test_begin()
        else:
            self.on_predict_begin()

    def _call_end_hook(self, mode):
        """Helper function for on_{train|test|predict}_end methods."""
        if mode == _TRAIN:
            self.on_train_end()
        elif mode == _TEST:
            self.on_test_end()
        else:
            self.on_predict_end()

    @torch.no_grad()
    def on_batch_begin(self, batch, logs=None):
        self._call_batch_hook(_TRAIN, 'begin', batch, logs=logs)

    @torch.no_grad()
    def on_batch_end(self, batch, logs=None):
        self._call_batch_hook(_TRAIN, 'end', batch, logs=logs)

    @torch.no_grad()
    def on_epoch_begin(self, epoch, logs=None):
        """Calls the `on_epoch_begin` methods of its callbacks.
        This function should only be called during train mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict, Currently no data is passed to this argument for this method
                but that may change in the future.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
        self._reset_batch_timing()

    @torch.no_grad()
    def on_epoch_end(self, epoch, logs=None):
        """Calls the `on_epoch_end` methods of its callbacks.
        This function should only be called during train mode.

        Arguments:
            epoch: integer, index of epoch.
            logs: dict, metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result keys
                are prefixed with `val_`.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    @torch.no_grad()
    def on_train_batch_begin(self, batch, logs=None):
        """Calls the `on_train_batch_begin` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        self._call_batch_hook(_TRAIN, 'begin', batch, logs=logs)

    @torch.no_grad()
    def on_train_batch_end(self, batch, logs=None):
        """Calls the `on_train_batch_end` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        self._call_batch_hook(_TRAIN, 'end', batch, logs=logs)

    @torch.no_grad()
    def on_test_batch_begin(self, batch, logs=None):
        """Calls the `on_test_batch_begin` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        self._call_batch_hook(_TEST, 'begin', batch, logs=logs)

    @torch.no_grad()
    def on_test_batch_end(self, batch, logs=None):
        """Calls the `on_test_batch_end` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        self._call_batch_hook(_TEST, 'end', batch, logs=logs)

    @torch.no_grad()
    def on_predict_batch_begin(self, batch, logs=None):
        """Calls the `on_predict_batch_begin` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, has keys `batch` and `size` representing the current
                batch number and the size of the batch.
        """
        self._call_batch_hook(_PREDICT, 'begin', batch, logs=logs)

    @torch.no_grad()
    def on_predict_batch_end(self, batch, logs=None):
        """Calls the `on_predict_batch_end` methods of its callbacks.

        Arguments:
            batch: integer, index of batch within the current epoch.
            logs: dict, metric results for this batch.
        """
        self._call_batch_hook(_PREDICT, 'end', batch, logs=logs)

    @torch.no_grad()
    def on_train_begin(self, logs=None):
        """Calls the `on_train_begin` methods of its callbacks.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    @torch.no_grad()
    def on_train_end(self, logs=None):
        """Calls the `on_train_end` methods of its callbacks.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_train_end(logs)

    @torch.no_grad()
    def on_test_begin(self, logs=None):
        """Calls the `on_test_begin` methods of its callbacks.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    @torch.no_grad()
    def on_test_end(self, logs=None):
        """Calls the `on_test_end` methods of its callbacks.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_test_end(logs)

    @torch.no_grad()
    def on_predict_begin(self, logs=None):
        """Calls the `on_predict_begin` methods of its callbacks.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_predict_begin(logs)

    @torch.no_grad()
    def on_predict_end(self, logs=None):
        """Calls the `on_predict_end` methods of its callbacks.

        Arguments:
            logs: dict, currently no data is passed to this argument for this method
                but that may change in the future.
        """
        for callback in self.callbacks:
            callback.on_predict_end(logs)

    def __iter__(self):
        return iter(self.callbacks)


