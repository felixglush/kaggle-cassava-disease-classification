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
from loss_functions import LabelSmoothingLoss, BiTemperedLoss
from pytorch_lightning.metrics.functional import confusion_matrix

from utils import average_model_state


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
        if finetune_model_fnames: print('Models to fine tune\n', self.finetune_model_fnames)
        self.freeze_bn = freeze_bn
        self.freeze_feature_extractor = freeze_feature_extractor

        self.holdout_df = holdout_df  # labelled if not kaggle, could be unlabelled test if kaggle
        self.config = config
        self.folds_df = folds_df
        self.current_fold = None
        self.lit_model = None
        self.lit_trainer = None
        self.criterion = None
        self.model_checkpoint_path = None
        self.checkpoint_params = checkpoint_params
        self.kaggle = kaggle

    def run(self):
        self.config.lr = 0.225
        assert self.folds_df is not None, 'Folds dataframe None'
        print(f'folds_df len {len(self.folds_df)}, holdout_df len {len(self.holdout_df)}')

        #starting_fold = self.get_restart_params()  # defaults to 0 and None checkpoint filepath.
        end_fold = self.config.fold_num  # this should be equal to len(self.finetuning_model_fnames) but just for sanity...

        if self.finetune_model_fnames:
            assert len(self.finetune_model_fnames) != self.config.fold_num, \
                print('NUMBER OF CHECKPOINTS DOESNT MATCH NUMBER OF FOLDS. SOMETHING COULD BE WRONG.')
            end_fold = len(self.finetune_model_fnames)

        self.data_module = LightningData(folds_df=self.folds_df, holdout_df=self.holdout_df, config=self.config)

        # self.criterion = LabelSmoothingLoss(num_classes=self.config.num_classes, smoothing=self.config.smoothing)
        # t1=0.3, t2=1.0 large margin noise (outliers far from decision boundary)
        # t1=1.0, t2=4.0 small margin noise (outliers close to decision boundary)
        self.criterion = BiTemperedLoss(smoothing=self.config.smoothing, t1=self.config.t1, t2=self.config.t2,
                                        num_classes=self.config.num_classes)

        model_args = {
            'config': self.config,
            'criterion': self.criterion,
            'pretrained': True,
            'lr': self.config.lr,
            'bn': self.freeze_bn,
            'features': self.freeze_feature_extractor
        }

        trainer_args = {
            'limit_train_batches': 1.0 if not self.config.debug else 2,  # for debug purposes
            'limit_val_batches': 1.0 if not self.config.debug else 2,
            'fast_dev_run': False if not self.config.debug else True,
            'accumulate_grad_batches': 4 if self.freeze_bn else 1,
            'enable_pl_optimizer': True,
            'gradient_clip_val': 1.5,
            'log_every_n_steps': 10,
            'flush_logs_every_n_steps': 25,
            # will be none after first resume
            'resume_from_checkpoint': self.model_checkpoint_path,
            'auto_lr_find': False if self.model_checkpoint_path or not self.config.lr_test else True,
            'benchmark': True,
            'default_root_dir': self.experiment_dir,
            'precision': 16,
            'gpus': 1,
            'min_epochs': self.config.epochs,
            'max_epochs': self.config.epochs * 4,
            'track_grad_norm': 2,
            'logger': pl_loggers.TensorBoardLogger(f'./runs/{self.experiment_name}'),
            # 'profiler': 'simple'
        }

        for fold in range(1, end_fold):
            print('Training fold', fold)
            self.current_fold = fold

            self.data_module.setup(str(fold))

            checkpoint = ModelCheckpoint(
                filename=f'{self.experiment_dir}/{self.config.model_arch}_bitempered_smooth={self.config.smoothing:.2f}' +
                         '_{val_loss:.3f}_{val_acc:.3f}_' + f'fold{fold}',
                monitor='val_loss',
                mode='min',
                save_top_k=1,
                verbose=True)
            early_stop = EarlyStopping(monitor='val_loss',
                                       patience=self.config.patience,
                                       verbose=True,
                                       mode='min')
            lr_monitor = LearningRateMonitor('step')

            model_args['len_trainloader'] = len(self.data_module.train_dataloader())
            trainer_args['callbacks'] = [checkpoint, early_stop, lr_monitor]

            self.run_fold(model_args, trainer_args)

            # we've resumed training by now
            self.model_checkpoint_path = None
            trainer_args['resume_from_checkpoint'] = None

    def get_restart_params(self):
        starting_fold = 0
        if self.checkpoint_params:
            if 'restart_from' in self.checkpoint_params and 'checkpoint_file_path' in self.checkpoint_params:
                starting_fold = self.checkpoint_params['restart_from']
                print('Restarting from fold', starting_fold, self.checkpoint_params['checkpoint_file_path'])
                self.model_checkpoint_path = self.checkpoint_params['checkpoint_file_path']
        return starting_fold

    def run_fold(self, model_args, trainer_args):
        """
            Trains a model with a particular fold
        """
        self.lit_trainer = Trainer(**trainer_args)
        self.lit_model = LightningModel(**model_args)

        if trainer_args['resume_from_checkpoint'] and not self.finetune:
            self.lit_model.load_from_checkpoint(trainer_args['resume_from_checkpoint'], config=self.config,
                                                criterion=self.criterion)
        elif self.finetune:
            checkpoint_filename = self.finetune_model_fnames[self.current_fold]
            print('Tuning', checkpoint_filename)
            # load the weights, not the state. we're training from epoch 0.
            self.lit_model.load_state_dict(state_dict=torch.load(checkpoint_filename)['state_dict'])

        self.lit_trainer.tune(model=self.lit_model, datamodule=self.data_module)  # optimal LR or max batch search
        self.lit_trainer.fit(model=self.lit_model, datamodule=self.data_module)

    def test(self, tta, weight_avg, mode='vote'):
        """ Loads all models and runs ensemble voting on holdout """
        self.test_data_module = LightningData(folds_df=None, holdout_df=self.holdout_df,
                                              config=self.config, kaggle=self.kaggle)
        self.test_data_module.setup('ensemble_holdout' if not self.kaggle else 'ensemble_test')

        testing_model = LightningModel(config=self.config, criterion=None, tta=tta)
        lit_tester = Trainer(
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
        if weight_avg:
            avg_state_dict = average_model_state(model_filenames, self.experiment_dir)
            self.test_model(all_model_predictions, avg_state_dict, lit_tester, testing_model)
            self.final_test_predictions = np.array(all_model_predictions).flatten()
        else:
            for filename in model_filenames:
                ckpt = torch.load(filename)
                self.test_model(all_model_predictions, ckpt['state_dict'], lit_tester, testing_model)
            all_model_predictions = np.array(all_model_predictions)

            if mode == 'avg':
                self.final_test_predictions = np.round(np.mean(all_model_predictions, axis=0), decimals=0)
            elif mode == 'vote':
                self.final_test_predictions = stats.mode(all_model_predictions, axis=0)[0].flatten()

        if not self.kaggle:
            self.test_confusion_matrix = confusion_matrix(
                preds=torch.tensor(self.final_test_predictions, dtype=torch.int),
                target=torch.tensor(self.holdout_df.label.values, dtype=torch.int),
                num_classes=self.config.num_classes,
                normalize='true')

    def test_model(self, all_model_predictions, state_dict, lit_tester, testing_model):
        testing_model.load_state_dict(state_dict=state_dict)
        lit_tester.test(testing_model, datamodule=self.test_data_module)
        all_model_predictions.append(testing_model.test_predictions)
