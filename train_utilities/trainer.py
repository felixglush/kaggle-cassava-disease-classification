import gc
import time

from pandas import DataFrame
import numpy as np
from sklearn.metrics import accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm

from common_utils import get_data_dfs, get_loaders, setup_model_optimizer, setup_scheduler_loss
from train_utilities.callbacks import CallbackGroupDelegator, EarlyStopping, ModelCheckpoint, MetricLogger, LRSchedule, \
    GradientHandler
from train_utilities.training_phase import Phase


def process_model_output(predictions, output, labels, num_correct):
    """ for each sample in this batch, take the maximum predicted class """
    batch_size = labels.size(0)
    max_prediction_per_sample = np.array([torch.argmax(output, 1).detach().cpu().numpy()])
    assert max_prediction_per_sample.shape == (1, batch_size)
    predictions = np.concatenate((predictions, max_prediction_per_sample), axis=None)
    num_correct += max_prediction_per_sample == labels.detach().cpu().numpy()
    return predictions, num_correct


class Trainer:
    def __init__(self, folds_df: DataFrame,
                 holdout_loader: DataLoader,
                 logger,
                 tensorboard_writer: SummaryWriter,
                 device,
                 settings,
                 checkpoint_params,
                 experiment_dir: str,
                 lr_test=False, clip=False, clip_at=1.0):
        self.clip_at = clip_at
        self.clip = clip
        self.current_gradient_norm = None
        self.experiment_dir = experiment_dir
        self.holdout_loader = holdout_loader
        self.checkpoint_params = checkpoint_params
        self.settings = settings
        self.folds_df = folds_df
        self.logger = logger
        self.tb_writer = tensorboard_writer
        self.device = device
        self.current_epoch = None
        self.current_fold = None
        self.stop = False
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.lr_test = lr_test
        self.model_checkpoint = None

        """
        Trainer manages loops over folds and over epochs so it stores metrics for...
        
        Metrics calculated after each fold:
            - holdout loss
            - holdout accuracy
            - predictions
        
        Metrics calculated after each epoch:
            - average train loss over all batches (mean of train phase losses)
            - average validation loss over all batches (mean of val phase losses)
            - validation accuracy (val phase accuracy)
            
        Metrics calculated during an epoch (stored in Phase):
            - current epoch losses 
            - current epoch accuracy
            - best val loss for current fold
        """
        self.metrics = {
            'fold_holdout_losses': [],
            'fold_holdout_accuracies': [],
            'fold_holdout_predictions': [],
            'epochs_avg_train_losses': [],
            'epochs_avg_val_losses': [],
            'epochs_val_accuracies': []
        }

        self.callbacksGroup = CallbackGroupDelegator([
            EarlyStopping(monitor='val_loss', logger=self.logger, patience=self.settings.loss_patience),
            ModelCheckpoint(directory=experiment_dir, logger=self.logger, metric='val_loss'),
            MetricLogger(logger=self.logger, tensorboard_writer=self.tb_writer),
            LRSchedule(),
            GradientHandler(self.settings.accum_iter)
        ])

    def fit(self):
        starting_fold = 0
        if self.checkpoint_params:
            # load
            if 'start_beginning_of' in self.checkpoint_params:
                starting_fold = self.checkpoint_params['start_beginning_of']
            elif 'restart_from' in self.checkpoint_params and 'checkpoint_file_path' in self.checkpoint_params:
                self.model_checkpoint = torch.load(self.checkpoint_params['checkpoint_file_path'])
                # best loss

        start_time = time.time()
        self.callbacksGroup.training_started()

        for fold in range(starting_fold, self.settings.fold_num):
            self.current_fold = fold
            self.run_fold()

        self.callbacksGroup.training_ended(elapsed_time=time.time() - start_time)

    def get_last_train_loss(self):
        return self.metrics['epochs_avg_train_losses'][-1]

    def get_last_val_loss(self):
        return self.metrics['epochs_avg_val_losses'][-1]

    def get_last_val_accuracy(self):
        return self.metrics['epochs_val_accuracies'][-1]

    def append_epochs_train_loss(self, loss):
        self.metrics['epochs_avg_train_losses'].append(loss)

    def append_epochs_val_loss(self, loss):
        self.metrics['epochs_avg_val_losses'].append(loss)

    def append_epochs_val_accuracy(self, accuracy):
        self.metrics['epochs_avg_val_accuracies'].append(accuracy)

    def append_fold_holdout_loss(self, loss):
        self.metrics['fold_holdout_losses'].append(loss)

    def append_fold_holdout_accuracy(self, accuracy):
        self.metrics['fold_holdout_accuracies'].append(accuracy)

    def append_fold_holdout_predictions(self, predictions):
        self.metrics['fold_holdout_predictions'].append(predictions)

    def run_fold(self):
        """
            Trains a new model corresponding to a particular fold over epochs
            Returns a DataFrame consisting of only the the rows used for validation along with corresponding predictions
        """
        model_checkpoint_name = self.experiment_dir + f'/{self.settings.model_arch}_fold{self.current_fold}.pth'

        # -------- DATASETS AND LOADERS --------
        # select the fold, create train & validation loaders
        train_df, valid_df = get_data_dfs(self.folds_df, self.current_fold)
        train_loader, valid_loader = get_loaders(train_df, valid_df, self.settings.train_bs,
                                                 self.settings.data_dir + '/train_images')

        learning_phases = [
            Phase('train', train_loader),
            Phase('valid', valid_loader, gradients=False),
        ]
        holdout_phase = Phase('holdout', self.holdout_loader, gradients=False, every_epoch=False)

        # make model and optimizer
        self.model, self.optimizer = setup_model_optimizer(model_arch=self.settings.model_arch,
                                                           lr=self.settings.lr,
                                                           is_amsgrad=self.settings.is_amsgrad,
                                                           num_labels=self.settings.num_classes,
                                                           weight_decay=self.settings.weight_decay,
                                                           momentum=self.settings.momentum,
                                                           fc_nodes=0,
                                                           device=self.device,
                                                           checkpoint=self.model_checkpoint)

        self.scheduler, self.criterion = setup_scheduler_loss(self.optimizer, self.lr_test, self.settings.verbose)
        grad_scaler = GradScaler()

        self.callbacksGroup.fold_started(fold=self.current_fold, total_folds=self.settings.fold_num)

        for e in range(self.settings.epochs):
            epoch_start_time = time.time()
            self.current_epoch = e

            self.callbacksGroup.epoch_started(epoch=e, total_epochs=self.settings.epochs)

            for phase in learning_phases:
                phase.reset()
                if phase.is_training:
                    avg_loss = self.train_epoch(phase, grad_scaler)
                    self.append_epochs_train_loss(avg_loss)
                else:
                    avg_loss, val_accuracy, _ = self.valid_epoch(phase)
                    self.append_epochs_val_loss(avg_loss)
                    self.append_epochs_val_accuracy(val_accuracy)
                # model checkpoint
                self.callbacksGroup.phase_ended(trainer=self, phase=phase)

            # epoch logging
            self.callbacksGroup.epoch_ended(metrics={
                'fold': self.current_fold,
                'epoch': e,
                'avg_train_loss': self.get_last_train_loss(),
                'avg_val_loss': self.get_last_val_loss(),
                'val_accuracy': self.get_last_val_accuracy(),
                'epoch_elapsed_time': time.time() - epoch_start_time
            })

            if self.stop:
                self.callbacksGroup.training_interrupted(
                    msg='Trainer received termination signal due to early stopping.')
                break

        checkpoint = torch.load(model_checkpoint_name)

        self.model.load_state_dict(checkpoint['model_state'])
        holdout_loss, holdout_preds, holdout_accuracy = self.valid_epoch(phase=holdout_phase)
        self.append_fold_holdout_loss(holdout_loss)
        self.append_fold_holdout_accuracy(holdout_accuracy)
        self.append_fold_holdout_predictions(holdout_preds)

        self.callbacksGroup.fold_ended(fold=self.current_fold, best_val_accuracy=checkpoint['accuracy'],
                                       holdout_accuracy=holdout_accuracy, holdout_loss=holdout_loss)

        del self.model, self.optimizer, train_loader, valid_loader

    def train_epoch(self, phase: Phase, grad_scaler: GradScaler):
        self.model.train()
        total_batches = len(phase.loader)
        progress_bar = tqdm(enumerate(phase.loader), total=total_batches)

        for batch_idx, (images, labels) in progress_bar:
            progress_bar.set_description(f"[TRAIN] Processing batch {batch_idx + 1}")
            phase.batch_idx = batch_idx

            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast():
                predictions = self.model(images)
                loss = self.criterion(predictions, labels)

            loss = loss / self.settings.accum_iter  # loss is normalized across the accumulated batches
            scaled_loss = grad_scaler.scale(loss)
            phase.update_loss(scaled_loss.detach().item())

            # See https://pytorch.org/docs/stable/amp.html#gradient-scaling for why scaling is helpful
            scaled_loss.backward()

            # do  gradient accumulation, scaler steps, clipping, etc
            self.callbacksGroup.after_backward_pass(trainer=self, scaler=grad_scaler)

            # update LR (if scheduler requires it), write logs
            self.callbacksGroup.batch_ended(trainer=self, phase=phase, wait_period=self.settings.wait_epochs_schd)

            gc.collect()
        return phase.average_epoch_loss()

    def valid_epoch(self, phase):
        self.model.eval()
        predictions = np.array([])
        num_correct, total = 0, 0
        total_batches = len(phase.loader)
        progress_bar = tqdm(enumerate(phase.loader), total=total_batches)

        for batch_idx, (images, labels) in progress_bar:
            phase.batch_idx = batch_idx
            progress_bar.set_description(f"[VAL] Processing batch {phase.batch_idx + 1}")

            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                output = self.model(images)  # output: [batch_size, # classes]

            loss = self.criterion(output, labels)
            phase.update_loss(loss.detach().item())

            # for each sample in this batch, take the maximum predicted class
            predictions, num_correct = process_model_output(predictions, output, labels, num_correct)
            total += labels.size(0)  # size of batch
            phase.accuracy = num_correct / total

            # logs
            self.callbacksGroup.batch_ended(trainer=self, phase=phase)

            gc.collect()

        phase.best_loss = min(phase.best_loss, phase.average_epoch_loss())
        phase.latest_preds = predictions
        return phase.average_epoch_loss(), phase.accuracy, phase.latest_preds
