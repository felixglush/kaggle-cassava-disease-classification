from pandas import DataFrame
import numpy as np
import torch
from config import Configuration
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GPUStatsMonitor

from lightning_objects import LightningModel, LightningData


class TrainManager:
    def __init__(self, folds_df: DataFrame,
                 holdout_df: DataFrame,
                 config: Configuration,
                 checkpoint_params,
                 experiment_dir: str, experiment_name: str):
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.holdout_df = holdout_df
        self.config = config
        self.folds_df = folds_df
        self.current_fold = None
        self.model = None
        self.criterion = None
        self.lr_test = config.lr_test
        self.model_checkpoint = None
        self.checkpoint_params = checkpoint_params

    def run(self):
        starting_fold = 0
        if self.checkpoint_params:  # TODO: finish implementation
            # load
            if 'start_beginning_of' in self.checkpoint_params:
                starting_fold = self.checkpoint_params['start_beginning_of']
            elif 'restart_from' in self.checkpoint_params and 'checkpoint_file_path' in self.checkpoint_params:
                self.model_checkpoint = torch.load(self.checkpoint_params['checkpoint_file_path'])

        self.data_module = LightningData(self.folds_df, holdout_df=self.holdout_df, config=self.config)

        self.criterion = torch.nn.CrossEntropyLoss()  # TODO: try different losses

        for fold in range(starting_fold, self.config.fold_num):
            self.current_fold = fold

            self.run_fold()

        self.run_ensemble()

    def run_fold(self):
        """
            Trains a new model corresponding to a particular fold
        """
        self.data_module.setup(str(self.current_fold))

        self.model = LightningModel(config=self.config, criterion=self.criterion, fold=self.current_fold,
                                    pretrained=True, lr=0.0001)

        checkpoint = ModelCheckpoint(filename=f'{self.experiment_dir}/{self.config.model_arch}_fold{self.current_fold}',
                                     monitor='val_loss',
                                     mode='min',
                                     save_top_k=1,
                                     verbose=True)

        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=self.config.patience,
                                   verbose=True,
                                   mode='min')
        gpu_stats = GPUStatsMonitor()

        trainer = Trainer(
            #limit_train_batches=10,
            #limit_val_batches=10,
            accumulate_grad_batches=self.config.grad_accumulator_steps,
            enable_pl_optimizer=True,
            auto_scale_batch_size='binsearch',
            auto_lr_find=True,
            benchmark=True,
            deterministic=True,
            default_root_dir=self.experiment_dir,
            precision=16,  # Halves the memory usage allowing for larger batches
            fast_dev_run=self.config.debug,
            gpus=1,
            callbacks=[checkpoint, early_stop, gpu_stats],
            min_epochs=min(self.config.epochs, 10),
            max_epochs=self.config.epochs,
            track_grad_norm=2,
            logger=pl_loggers.TensorBoardLogger(f'./runs/{self.experiment_name}'))

        # tunes LR and finds max batch_size that will fit in memory
        trainer.tune(model=self.model, datamodule=self.data_module)

        trainer.fit(model=self.model, datamodule=self.data_module)

        self.data_module.setup('holdout')
        trainer.test(model=self.model, datamodule=self.data_module)

    def run_ensemble(self):
        """ Loads all models and runs ensemble voting on holdout """


def process_model_output(predictions, output, labels, num_correct):
    """ for each sample in this batch, take the maximum predicted class """
    batch_size = labels.size(0)
    max_prediction_per_sample = np.array([torch.argmax(output, 1).detach().cpu().numpy()])
    assert max_prediction_per_sample.shape == (1, batch_size)
    predictions = np.concatenate((predictions, max_prediction_per_sample), axis=None)
    num_correct += max_prediction_per_sample == labels.detach().cpu().numpy()
    return predictions, num_correct
