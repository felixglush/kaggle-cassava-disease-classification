import glob

import numpy as np
import torch
from pandas import DataFrame
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, GPUStatsMonitor
from scipy.stats import stats

from config import Configuration
from lightning_objects import LightningModel, LightningData


class TrainManager:
    def __init__(self, holdout_df: DataFrame,
                 config: Configuration,
                 experiment_dir: str,
                 experiment_name: str,
                 checkpoint_params=None,
                 folds_df: DataFrame = None,
                 kaggle=False):
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name
        self.holdout_df = holdout_df  # could be labelled if not kaggle, could be test test if kaggle
        self.config = config
        self.folds_df = folds_df
        self.current_fold = None
        self.model = None
        self.criterion = None
        self.lr_test = config.lr_test
        self.model_checkpoint_path = None
        self.checkpoint_params = checkpoint_params
        self.kaggle = kaggle

    def run(self):
        assert self.folds_df is not None
        starting_fold = self.get_restart_params()  # defaults to 0 and None checkpoint filepath.

        self.data_module = LightningData(folds_df=self.folds_df, holdout_df=self.holdout_df, config=self.config)

        self.criterion = torch.nn.CrossEntropyLoss()  # TODO: try different losses

        for fold in range(starting_fold, self.config.fold_num):
            self.current_fold = fold
            print('Training fold', fold)
            self.run_fold()

    def get_restart_params(self):
        starting_fold = 0
        if self.checkpoint_params:
            if 'restart_from' in self.checkpoint_params and 'checkpoint_file_path' in self.checkpoint_params:
                starting_fold = self.checkpoint_params['restart_from']
                print('Restarting from fold', starting_fold)
                self.model_checkpoint_path = self.checkpoint_params['checkpoint_file_path']
        return starting_fold

    def run_fold(self):
        """
            Trains a new model corresponding to a particular fold
        """
        self.data_module.setup(str(self.current_fold))

        self.model = LightningModel(config=self.config, criterion=self.criterion,
                                    pretrained=True, lr=self.config.lr)

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
            # limit_train_batches=10,
            # limit_val_batches=5,
            accumulate_grad_batches=self.config.grad_accumulator_steps,  # TODO: turn off with BatchNorm
            enable_pl_optimizer=True,
            # auto_scale_batch_size='binsearch',
            auto_lr_find=True,
            benchmark=True,
            # deterministic=True,
            default_root_dir=self.experiment_dir,
            precision=16,
            fast_dev_run=self.config.debug,
            gpus=1,
            callbacks=[checkpoint, early_stop, gpu_stats],
            min_epochs=min(self.config.epochs, 10),
            max_epochs=self.config.epochs,
            track_grad_norm=2,
            resume_from_checkpoint=self.model_checkpoint_path,
            logger=pl_loggers.TensorBoardLogger(f'./runs/{self.experiment_name}'),
            # profiler='simple'
        )

        # tunes LR and finds max batch_size that will fit in memory
        trainer.tune(model=self.model, datamodule=self.data_module)

        trainer.fit(model=self.model, datamodule=self.data_module)

        self.data_module.setup('holdout')
        self.model.load_from_checkpoint(checkpoint_path=checkpoint.best_model_path)
        trainer.test(model=self.model, datamodule=self.data_module)

    def run_ensemble(self, mode='vote'):
        """ Loads all models and runs ensemble voting on holdout """
        self.ensemble_data_module = LightningData(folds_df=None, holdout_df=self.holdout_df,
                                                  config=self.config, kaggle=self.kaggle)
        self.ensemble_data_module.setup('ensemble_holdout' if not self.kaggle else 'ensemble_test')
        testing_model = LightningModel(config=self.config, criterion=None,
                                       pretrained=True, lr=self.config.lr)
        tester = Trainer(
            default_root_dir=self.experiment_dir,
            precision=16,
            fast_dev_run=False,
            gpus=1,
            logger=pl_loggers.TensorBoardLogger(f'./runs/{self.experiment_name}'),
        )

        # loop over models
        # each row holds predictions for each model
        # [[model 1  preds], [model 2 preds], ...]
        model_filenames = glob.glob(self.experiment_dir + '/*fold*.ckpt')
        all_model_predictions = []
        for filename in model_filenames:
            ckpt = torch.load(filename)
            testing_model.load_state_dict(state_dict=ckpt['state_dict'])  # assuming one architecture
            tester.test(testing_model, datamodule=self.ensemble_data_module)
            all_model_predictions.append(testing_model.test_predictions)

        all_model_predictions = np.array(all_model_predictions)

        print('all pred shape', all_model_predictions.shape)


        if mode == 'avg':
            self.test_predictions_ensembled = np.round(np.mean(all_model_predictions, axis=0), decimals=0).flatten()
        elif mode == 'vote':
            self.test_predictions_ensembled = stats.mode(all_model_predictions, axis=0)[0].flatten()
