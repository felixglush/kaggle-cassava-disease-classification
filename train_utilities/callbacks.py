import re
from datetime import timedelta
from typing import List

from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from train_utilities.trainer import Trainer
from train_utilities.training_phase import Phase


"""
This file defines a base class all callbacks inherit, a delegator, and the callbacks themselves.
Why callbacks? They let us make modular changes to the training code without rewriting the training loop itself.
"""


class CallbackBase:
    """ A base class that all callbacks inherit. """

    def fold_started(self, **kwargs): pass

    def fold_ended(self, **kwargs): pass

    def phase_started(self, **kwargs): pass

    def phase_ended(self, **kwargs): pass

    def training_started(self, **kwargs): pass

    def training_ended(self, **kwargs): pass

    def training_interrupted(self, **kwargs): pass

    def epoch_started(self, **kwargs): pass

    def epoch_ended(self, **kwargs): pass

    def batch_started(self, **kwargs): pass

    def batch_ended(self, **kwargs): pass

    def gradients_accumulated(self, **kwargs): pass

    def before_forward_pass(self, **kwargs): pass

    def after_forward_pass(self, **kwargs): pass

    def before_backward_pass(self, **kwargs): pass

    def after_backward_pass(self, **kwargs): pass


def to_snake_case(string):
    """Converts CamelCase string into snake_case."""
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()


def classname(obj):
    return obj.__class__.__name__


class CallbackGroupDelegator(CallbackBase):
    """
        This invokes callbacks. Example usage:

        cb = CallbacksGroup([
            EarlyStopping(),
            Checkpoint(),
            MetricLogger()
        ])

        Then we can call `cb.<function>(**kwargs)` and each callback  will call its implementation of <function>.
    """

    def __init__(self, callbacks: List):
        self.callbacks = callbacks
        # {"rolling_loss": RollingLoss(), etc}
        self.named_callbacks = {to_snake_case(classname(cb)): cb for cb in self.callbacks}

    def __getitem__(self, item):
        item = to_snake_case(item)
        if item in self.named_callbacks:
            return self.named_callbacks[item]
        raise KeyError(f'callback name is not found: {item}')

    def invoke(self, method, **kwargs):
        """ Calls each callback in the group with the supplied method """
        # 'method' is a string and we need the function attribute itself
        for cb in self.callbacks:
            # equivalent to cb.<method>(kwargs)
            getattr(cb, method)(**kwargs)

    def fold_started(self, **kwargs):
        self.invoke('fold_started', **kwargs)

    def fold_ended(self, **kwargs):
        self.invoke('fold_ended', **kwargs)

    def training_started(self, **kwargs):
        self.invoke('training_started', **kwargs)

    def training_ended(self, **kwargs):
        self.invoke('training_ended', **kwargs)

    def training_interrupted(self, **kwargs):
        self.invoke('training_interrupted', **kwargs)

    def epoch_started(self, **kwargs):
        self.invoke('epoch_started', **kwargs)

    def phase_started(self, **kwargs):
        self.invoke('phase_started', **kwargs)

    def phase_ended(self, **kwargs):
        self.invoke('phase_ended', **kwargs)

    def epoch_ended(self, **kwargs):
        self.invoke('epoch_ended', **kwargs)

    def batch_started(self, **kwargs):
        self.invoke('batch_started', **kwargs)

    def gradients_accumulated(self, **kwargs):
        self.invoke('gradients_accumulated', **kwargs)

    def batch_ended(self, **kwargs):
        self.invoke('batch_ended', **kwargs)

    def before_forward_pass(self, **kwargs):
        self.invoke('before_forward_pass', **kwargs)

    def after_forward_pass(self, **kwargs):
        self.invoke('after_forward_pass', **kwargs)

    def before_backward_pass(self, **kwargs):
        self.invoke('before_backward_pass', **kwargs)

    def after_backward_pass(self, **kwargs):
        self.invoke('after_backward_pass', **kwargs)


class EarlyStopping(CallbackBase):
    def __init__(self, monitor, logger, delta_improve=0, patience=5):
        self.monitor = monitor
        self.delta_improve = delta_improve
        self.counter = 0
        self.patience = patience
        self.logger = logger

    def phase_ended(self, trainer: Trainer, phase: Phase, **kwargs):
        if self.monitor == 'val_loss' and not phase.is_training:
            score_improved = phase.average_epoch_loss() >= phase.best_loss - self.delta_improve

            if score_improved:
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.logger.info(
                        f'Metric {self.monitor} has not seen improvement in {self.patience} epochs. Early stop.')
                    trainer.stop = True


class ModelCheckpoint(CallbackBase):
    """
    directory: directory to save the model in
    monitor: the metric to monitor for improvement. Model is saved whenever this is better than the best seen so far.
    """

    def __init__(self, directory, logger, metric='val_loss', verbose=True):
        self.directory = directory
        self.metric = metric
        self.logger = logger
        self.verbose = verbose and logger

    def phase_ended(self, trainer: Trainer, phase: Phase, **kwargs):
        if self.metric == 'val_loss' and not phase.is_training and\
                phase.average_epoch_loss() >= phase.best_loss:
            phase.best_loss = phase.average_epoch_loss()
            torch.save({'model_state': trainer.model.state_dict(),
                        'optimizer_state': trainer.optimizer.state_dict(),
                        'preds': phase.latest_preds,
                        'accuracy': phase.accuracy,
                        'val_loss': phase.best_loss,
                        'fold': trainer.current_fold,
                        'epoch_saved_at': trainer.current_epoch
                        },
                       f'{self.directory}/{trainer.settings.model_arch}_fold{trainer.current_fold}')
            if self.verbose:
                self.logger.info('Score improved - model saved')
        else:
            if self.verbose:
                self.logger.info('Score not improved - model not saved')


class MetricLogger(CallbackBase):
    def __init__(self, logger, tensorboard_writer: SummaryWriter):
        self.logger = logger
        self.tb_writer = tensorboard_writer

    def training_started(self, **kwargs):
        if self.logger:
            self.logger.info('Training started')

    def training_ended(self, elapsed_time, **kwargs):
        if self.logger:
            self.logger.info(f'Training complete in {str(timedelta(seconds=elapsed_time))}')

    def fold_started(self, fold, total_folds, **kwargs):
        if self.logger:
            self.logger(f'Training fold {fold}/{total_folds}')

    def fold_ended(self, fold, best_val_accuracy, holdout_accuracy, holdout_loss, **kwargs):
        if self.tb_writer:
            self.tb_writer.add_scalar(f'Fold {fold} best validation accuracy', best_val_accuracy, fold)
            self.tb_writer.add_scalar(f'Fold {fold} holdout accuracy', holdout_accuracy, fold)
            self.tb_writer.add_scalar(f'Fold {fold} holdout loss', holdout_loss, fold)
        if self.logger:
            self.logger.info(f'\nFold {fold} summary:\n ' + \
                             f'Best validation accuracy: {best_val_accuracy} ' + \
                             f'Holdout accuracy: {holdout_accuracy} ' + \
                             f'Holdout loss: {holdout_loss}')

    def epoch_started(self, **kwargs):
        pass

    def epoch_ended(self, metrics, **kwargs):
        fold = metrics.fold + 1
        avg_train_loss = metrics.avg_train_loss
        avg_val_loss = metrics.avg_val_loss
        val_accuracy = metrics.val_accuracy
        e = metrics.epoch

        if self.tb_writer:
            self.tb_writer.add_scalar(f'Avg Epoch Train Loss Fold {fold}', avg_train_loss, e)
            self.tb_writer.add_scalar(f'Avg Epoch Val Loss Fold {fold}', avg_val_loss, e)
            self.tb_writer.add_scalar(f'Epoch Val Accuracy Fold {fold}', val_accuracy, e)
        if self.logger:
            self.logger.info(f'\nEpoch training summary:\n Fold {fold}/{total_folds} | ' + \
                             f'Epoch: {e + 1}/{total_epochs} | ' + \
                             f'Epoch time: {metrics.epoch_elapsed_time} sec\n' + \
                             f'Training loss: {avg_train_loss} | ' + \
                             f'Validation loss: {avg_val_loss} | ' + \
                             f'Accuracy: {val_accuracy}')

    def phase_started(self, **kwargs):
        pass

    def phase_ended(self, **kwargs):
        pass

    def batch_started(self, **kwargs):
        pass

    def batch_ended(self, trainer: Trainer, phase: Phase, **kwargs):
        if self.tb_writer and (phase.batch_idx + 1) % trainer.settings.print_every == 0:
            total_batches_processed = trainer.current_epoch * len(phase.loader) + phase.batch_idx
            if phase.is_training:  # this was a training batch
                self.tb_writer.add_scalar(f'Train loss fold {trainer.current_fold}',
                                          phase.running_loss / (phase.batch_idx + 1),
                                          total_batches_processed)
                self.tb_writer.add_scalar(f'Gradient fold {trainer.current_fold}',
                                          trainer.current_gradient_norm,
                                          total_batches_processed)
                self.tb_writer.add_scalar(f'Learning Rate',
                                          trainer.optimizer.param_groups[0]['lr'],
                                          total_batches_processed)
            else:
                if phase.every_epoch:
                    self.tb_writer.add_scalar(f'Validation loss fold {trainer.current_fold}',
                                              phase.running_loss / (phase.batch_idx + 1),
                                              total_batches_processed)
                else:  #
                    self.tb_writer.add_scalar(f'Holdout loss fold {trainer.current_fold}',
                                              phase.running_loss / (phase.batch_idx + 1),
                                              trainer.current_fold)


class LRSchedule(CallbackBase):
    def __int__(self):
        pass

    def batch_ended(self, trainer: Trainer, wait_period, **kwargs):
        if trainer.scheduler and \
                ((trainer.lr_test and isinstance(trainer.scheduler, torch.optim.lr_scheduler.StepLR)) or
                 (trainer.current_epoch > wait_period and
                  isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts))):
            trainer.scheduler.step()


class GradientHandler(CallbackBase):
    def __init__(self, accumulate_batches):
        self.accumlate_batches = accumulate_batches

    def after_backward_pass(self, trainer: Trainer, scaler: GradScaler, phase: Phase, **kwargs):
        if (phase.batch_idx + 1) % self.accumlate_batches == 0 or (phase.batch_idx + 1) == len(phase.loader):
            scaler.unscale_(trainer.optimizer)

            # Monitor gradients for explosions
            total_norm = 0.0
            for p in list(filter(lambda param: param.grad is not None, trainer.model.parameters())):
                param_norm = p.grad.data.norm(2).item()  # norm of the gradient tensor
                total_norm += param_norm ** 2
            total_norm = np.sqrt(total_norm)
            trainer.current_gradient_norm = total_norm

            if trainer.clip:
                torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.clip_at)

            scaler.step(trainer.optimizer)
            scaler.update()
            trainer.optimizer.zero_grad()
