import time

from torch.cuda.amp import GradScaler

from common_utils import get_data_dfs, get_loaders, setup_model_optimizer, get_schd_crit
from early_stopping import EarlyStopping

from train_utilities.callbacks import CallbackGroupDelegator
from train_utilities.training_phase import Phase


class Trainer():
    def __init__(self, folds_df, holdout_loader, logger, tensorboard_writer,
                 device, settings, checkpoint_params, experiment_dir):
        self.experiment_dir = experiment_dir
        self.holdout_loader = holdout_loader
        self.checkpoint_params = checkpoint_params
        self.settings = settings
        self.folds_df = folds_df
        self.logger = logger
        self.tensorboard_writer = tensorboard_writer
        self.device = device
        self.epoch = None
        self.fold = None
        self.train_loader = None
        self.valid_loader = None
        self.holdout_loader = None
        self.model = None
        self.callbacksGroup = CallbackGroupDelegator([
            RollingLoss(),
            Score(),
            Scheduler(),
            OutputLogger(tb_writer)
        ])

    def fit(self):
        starting_fold = 0
        if self.checkpoint_params:
            # load
            self.checkpoint = None
            pass

        start_time = time.time()
        self.callbacksGroup.training_started(start_time=start_time)

        for fold in range(starting_fold, self.settings.fold_num):
            self.callbacksGroup.fold_started(fold=fold)
            self.fold = fold
            self.run_fold(fold)
            self.callbacksGroup.fold_ended(fold=fold)

        self.callbacksGroup.training_ended(elapsed_time=time.time() - start_time)

    def run_fold(self, fold):
        """
            Trains a new model corresponding to a particular fold over epochs
            Returns a DataFrame consisting of only the the rows used for validation along with corresponding predictions
        """
        model_checkpoint_name = self.experiment_dir + f'/{self.settings.model_arch}_fold{fold}.pth'

        # -------- DATASETS AND LOADERS --------
        # select the fold, create train & validation loaders
        train_df, valid_df = get_data_dfs(self.folds_df, fold)
        train_loader, valid_loader = get_loaders(train_df, valid_df, self.settings.train_bs,
                                                 self.settings.data_dir + '/train_images')

        phases = [
            Phase('train', train_loader),
            Phase('valid', valid_loader, gradients=False),
            Phase('holdout', self.holdout_loader, gradients=False, every_epoch=False)
        ]

        # make model and optimizer
        model, optimizer = setup_model_optimizer(model_arch=self.settings.model_arch,
                                                 lr=self.settings.lr,
                                                 is_amsgrad=self.settings.is_amsgrad,
                                                 num_labels=self.settings.num_classes,
                                                 weight_decay=self.settings.weight_decay,
                                                 momentum=self.settings.momentum,
                                                 fc_layer={"middle_fc": False, "middle_fc_size": 0},
                                                 device=self.device,
                                                 checkpoint=self.checkpoint)

        scheduler, criterion = get_schd_crit(optimizer)

        accuracy = 0.
        best_val_loss = float('inf')
        train_losses, val_losses = [], []

        early_stop = EarlyStopping('val_loss', self.logger, patience=self.settings.loss_patience)

        for e in range(self.settings.epochs):
            self.epoch = e
            epoch_start_time = time.time()

            self.callbacksGroup.epoch_started(epoch=e)

            for phase in phases:
                if phase.every_epoch:
                    batches = len(phase.loader)
                    is_training = phase.gradients
                    self.callbacksGroup.phase_started(phase=phase, total_batches=batches)

                    if is_training:
                        loss = self.train_epoch(train_loader, model, criterion, optimizer, scheduler,
                                                             GradScaler(), fold)
                    else:
                        loss, preds = self.valid_epoch(valid_loader, model, criterion, fold)

                    phase.batch_loss = loss
                    self.callbacksGroup.phase_ended(phase=phase, loss=loss, preds=preds)

            self.callbacksGroup.epoch_ended(epoch=e)
            ########################################################################################################################

            # -------- SCORE METRICS & LOGGING FOR THIS EPOCH --------
            validation_labels = valid_df[self.settings.target_col].values
            accuracy = accuracy_score(y_true=validation_labels, y_pred=preds)

            epoch_elapsed_time = time.time() - epoch_start_time

            tb_writer.add_scalar(f'Avg Epoch Train Loss Fold {fold}', avg_training_loss, e)
            tb_writer.add_scalar(f'Avg Epoch Val Loss Fold {fold}', avg_validation_loss, e)
            tb_writer.add_scalar(f'Epoch Val Accuracy Fold {fold}', accuracy, e)

            LOGGER.info(f'\nEpoch training summary:\n Fold {fold + 1}/{self.settings.fold_num} | ' + \
                        f'Epoch: {e + 1}/{self.settings.epochs} | ' + \
                        f'Epoch time: {epoch_elapsed_time} sec\n' + \
                        f'Training loss: {avg_training_loss} | ' + \
                        f'Validation loss: {avg_validation_loss} | ' + \
                        f'Accuracy: {accuracy}')

            early_stop(avg_validation_loss)
            if early_stop.stop: break

            # --------SAVE MODEL --------
            if avg_validation_loss < best_val_loss:
                best_val_loss = avg_validation_loss
                torch.save({'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'accuracy': accuracy,
                            'preds': preds,
                            'val_loss': best_val_loss,
                            'fold': fold,
                            'epochs_no_improve': early_stop.counter,
                            'epoch_stopped_at': e
                            }, model_checkpoint_name)
                LOGGER.info(f'Saved model!')
            LOGGER.info('----------------')

            # -------- UPDATE LR --------
            if scheduler and e > 2:
                if self.settings.scheduler == 'ReduceLROnPlateau':
                    scheduler.step(avg_validation_loss)
            gc.collect()

        # -------- TEST ON HOLDOUT SET --------
        # load best model
        checkpoint = torch.load(model_checkpoint_name)
        model.load_state_dict(checkpoint['model_state'])
        holdout_loss, holdout_preds = valid_epoch(holdout_dataloader, model,
                                                  criterion, LOGGER, device,
                                                  tb_writer, fold, holdout=True)
        holdout_accuracy = accuracy_score(y_true=holdout_targets, y_pred=holdout_preds)

        tb_writer.add_scalar(f'Fold {fold} holdout accuracy', holdout_accuracy, fold)
        tb_writer.add_scalar(f'Fold {fold} holdout loss', holdout_loss, fold)

        valid_df['prediction'] = checkpoint['preds']

        del model
        del optimizer
        del train_loader
        del valid_loader
        return valid_df, checkpoint['accuracy'], holdout_accuracy
