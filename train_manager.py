import glob

import numpy as np
import torch
from pandas import DataFrame
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from scipy.stats import stats

from config import Configuration
from lightning_objects import LightningModel, LightningData


class TrainManager:
    def __init__(self, holdout_df: DataFrame,
                 config: Configuration,
                 experiment_dir: str,
                 experiment_name: str,
                 finetune,
                 freeze_bn,
                 freeze_feature_extractor,
                 finetune_model_fnames=None,
                 checkpoint_params=None,
                 folds_df: DataFrame = None,
                 kaggle=False):
        self.experiment_dir = experiment_dir
        self.experiment_name = experiment_name

        self.finetune = finetune
        self.finetune_model_fnames = finetune_model_fnames
        if finetune_model_fnames: print('Models to fine tune', self.finetune_model_fnames, '\n')
        self.freeze_bn = freeze_bn
        self.freeze_feature_extractor = freeze_feature_extractor

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
        self.config.train_bs = 16
        assert self.folds_df is not None
        print(f'folds_df len {len(self.folds_df)}, holdout_df len {len(self.holdout_df)}')

        starting_fold = self.get_restart_params()  # defaults to 0 and None checkpoint filepath.
        end_fold = self.config.fold_num  # this should be equal to len(self.finetuning_model_fnames) but just for sanity...

        if self.finetune_model_fnames and len(self.finetune_model_fnames) != self.config.fold_num:
            print('NUMBER OF CHECKPOINTS DOESNT MATCH NUMBER OF FOLDS. SOMETHING COULD BE WRONG.')
        if self.finetune_model_fnames:
            end_fold = len(self.finetune_model_fnames)

        self.data_module = LightningData(folds_df=self.folds_df, holdout_df=self.holdout_df, config=self.config)

        self.criterion = torch.nn.CrossEntropyLoss()  # TODO: try different losses

        for fold in range(1, end_fold):
            self.current_fold = fold
            print('Training fold', fold)
            self.run_fold()

    def get_restart_params(self):
        starting_fold = 0
        if self.checkpoint_params:
            if 'restart_from' in self.checkpoint_params and 'checkpoint_file_path' in self.checkpoint_params:
                starting_fold = self.checkpoint_params['restart_from']
                print('Restarting from fold', starting_fold, self.checkpoint_params['checkpoint_file_path'])
                self.model_checkpoint_path = self.checkpoint_params['checkpoint_file_path']
        return starting_fold

    def run_fold(self):
        """
            Trains a model corresponding to a particular fold
        """
        self.data_module.setup(str(self.current_fold))

        self.model = LightningModel(config=self.config, criterion=self.criterion,
                                    pretrained=True, lr=0.22,
                                    bn=self.freeze_bn, features=self.freeze_feature_extractor)

        if self.finetune:
            ckpt_fname = self.finetune_model_fnames[self.current_fold]
            print('Tuning', ckpt_fname)
            self.model.load_state_dict(state_dict=torch.load(ckpt_fname)['state_dict'])
            #self.model.load_from_checkpoint(ckpt_fname, config=self.config, criterion=self.criterion)
        for name, param in self.model.named_parameters():
            print(name, param.requires_grad)

        checkpoint = ModelCheckpoint(filename=f'{self.experiment_dir}/{self.config.model_arch}_fold{self.current_fold}',
                                     monitor='val_loss',
                                     mode='min',
                                     save_top_k=1,
                                     verbose=True)

        early_stop = EarlyStopping(monitor='val_loss',
                                   patience=self.config.patience,
                                   verbose=True,
                                   mode='min')

        lr_monitor = LearningRateMonitor('step')

        trainer = Trainer(
            fast_dev_run=self.config.debug,
            accumulate_grad_batches=4 if self.freeze_bn else 1,
            enable_pl_optimizer=True,
            gradient_clip_val=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=50,
            #auto_lr_find=True,
            #auto_scale_batch_size='binsearch', # may be too large if just training the classifier
            benchmark=True,
            # deterministic=True,
            default_root_dir=self.experiment_dir,
            precision=16,
            gpus=1,
            callbacks=[checkpoint, early_stop, lr_monitor],
            min_epochs=min(self.config.epochs, 10),
            max_epochs=self.config.epochs,
            track_grad_norm=2,
            resume_from_checkpoint=self.model_checkpoint_path,
            logger=pl_loggers.TensorBoardLogger(f'./runs/{self.experiment_name}'),
            # profiler='simple'
        )

        # lr_finder = trainer.tuner.lr_find(self.model, datamodule=self.data_module)
        # suggested_lr = lr_finder.suggestion()
        # print('Suggested LR', suggested_lr)
        # self.model.hparams.lr = suggested_lr
        trainer.tune(model=self.model, datamodule=self.data_module) # optimal LR or max batch search

        trainer.fit(model=self.model, datamodule=self.data_module)

        self.data_module.setup('holdout')
        self.model.load_from_checkpoint(checkpoint_path=checkpoint.best_model_path, config=self.config, criterion=self.criterion)
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
