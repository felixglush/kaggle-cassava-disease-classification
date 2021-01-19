import sys

import torch
from pandas import DataFrame
from torch.nn import Dropout, Sequential, Linear
from torch.optim import SGD, Optimizer, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from adabound import AdaBound
from cassava_dataset import CassavaDataset
from utils import get_train_transforms, get_valid_transforms, get_tta_transforms
from config import Configuration

sys.path.append('../pytorch-image-models')
import timm  # pytorch-image-models implementations

from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics.functional.classification import accuracy
from typing import Optional, Union, List, Any, Dict, Callable

import numpy as np


# EfficientNet noisy student: https://arxiv.org/pdf/1911.04252.pdf.
# Implementation from https://github.com/rwightman/pytorch-image-models.
class LightningModel(LightningModule):
    def __init__(self, config: Configuration, criterion, len_trainloader=0, lr=0.1,
                 fc_nodes=0, pretrained=False, bn=True, features=False, tta=0):
        super().__init__()
        self.len_dataloader = len_trainloader
        self.valid_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.test_predictions = []
        self.config = config
        self.criterion = criterion
        self.lr = lr
        self.tta = tta
        self.num_correct = 0
        self.seen = 0
        self.model = timm.create_model(config.model_arch, pretrained=pretrained)
        if config.model_arch.find('efficient') >= 0:
            self.model.classifier = Sequential(
                Dropout(p=0.25),
                Linear(self.model.classifier.in_features, self.config.num_classes)
            )
        elif config.model_arch == 'seresnet50':
            self.model.fc = Sequential(
                Dropout(p=0.3),
                torch.nn.Linear(self.model.fc.in_features, self.config.num_classes)
            )
        else:
            print('unsupported model architecture:', config.model_arch)
        self.freeze_layers(bn, features)
        self.save_hyperparameters()

    def freeze_layers(self, bn, features):
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.modules.Linear):
                for p in module.parameters():
                    p.requires_grad = False if features else True

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
                for p in module.parameters():
                    p.requires_grad = False if bn else True

        # for name, p in self.model.named_parameters():
        #     print(name, p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr)
        # opt = SGD(self.parameters(), lr=self.lr, momentum=self.config.momentum, nesterov=True)
        # opt = Adam(self.model.parameters(), self.lr,
        #            weight_decay=self.config.weight_decay, amsgrad=self.config.is_amsgrad)
        # opt = AdaBound(self.model.parameters(), self.lr, gamma=1e-2)

        scheduler = OneCycleLR(opt,
                               max_lr=self.lr,
                               steps_per_epoch=self.len_dataloader,
                               epochs=self.config.epochs * 2)
        # scheduler = CosineAnnealingWarmRestarts(opt, T_0=self.len_dataloader,
        #                                         T_mult=self.config.T_mult,
        #                                         eta_min=self.config.min_lr)
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        labels, loss, predictions = self.label_forward_pass(batch)
        # https://stats.stackexchange.com/questions/218656/classification-with-noisy-labels
        noise = 0.05
        predictions = noise / self.config.num_classes + (1-noise) * predictions

        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': loss, 'preds': predictions, 'target': labels}

    def validation_step(self, batch, batch_idx):
        labels, loss, predictions = self.label_forward_pass(batch)
        self.log('val_loss', loss, on_epoch=True, logger=True, prog_bar=True)
        return {'loss': loss, 'preds': predictions, 'target': labels}

    def validation_step_end(self, outputs):
        self.valid_accuracy(outputs['preds'], outputs['target'])

    def on_validation_epoch_end(self) -> None:
        self.log('val_acc', self.valid_accuracy.compute(), prog_bar=True, logger=True)
        self.valid_accuracy.reset()

    def label_forward_pass(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        return labels, loss, predictions

    def on_test_epoch_start(self) -> None:
        self.test_accuracy.reset()
        self.test_predictions = []

    def test_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self.model(images)
        most_likely = torch.argmax(predictions, dim=1)  # highest preds for each sample in batch
        return {'preds': most_likely, 'target': labels}

    def test_step_end(self, outputs):
        self.test_predictions.extend(outputs['preds'].cpu().numpy())
        self.test_accuracy(outputs['preds'], outputs['target'])

    def on_test_epoch_end(self) -> None:
        print('Test epoch ended.')
        self.log('test_acc', self.test_accuracy.compute(), prog_bar=True, logger=True)


class LightningData(LightningDataModule):

    def __init__(self, holdout_df: DataFrame, config: Configuration, folds_df: DataFrame = None, kaggle=False, tta=0):
        super().__init__()
        self.folds_df = folds_df
        self.holdout_df = holdout_df
        self.config = config
        self.batch_size = config.train_bs
        self.fold_range = [str(i) for i in range(self.config.fold_num)]
        self.train_transforms = get_train_transforms(self.config.img_size)
        self.valid_transforms = get_valid_transforms(self.config.img_size)
        self.test_transforms = get_tta_transforms(self.config.img_size) if tta > 0 else self.valid_transforms
        self.kaggle = kaggle
        self.tta = tta

    def setup(self, stage: Optional[str] = None):
        if stage == 'holdout' or stage == 'ensemble_holdout' or stage == 'ensemble_test':
            self.test_loader = CassavaDataset(self.holdout_df, self.config.train_img_dir,
                                              output_label=(not self.kaggle),
                                              transform=self.test_transforms)
        elif stage in self.fold_range:
            self.get_data_dfs_from_fold(self.folds_df, int(stage))
            self.create_weighted_sampler(self.train_df)

            self.train_dataset = CassavaDataset(self.train_df, self.config.train_img_dir,
                                                output_label=True, transform=self.train_transforms)
            self.valid_dataset = CassavaDataset(self.valid_df, self.config.train_img_dir,
                                                output_label=True, transform=self.valid_transforms)

    def get_data_dfs_from_fold(self, df: DataFrame, fold: int):
        train_idx = df[df['fold'] != fold].index
        valid_idx = df[df['fold'] == fold].index
        # since we are selecting rows, the index will be non contiguous #s so reset
        self.train_df = df.iloc[train_idx].reset_index(drop=True)
        self.valid_df = df.iloc[valid_idx].reset_index(drop=True)

    def create_weighted_sampler(self, train_df):
        train_target = train_df.label.values
        class_sample_count = np.unique(train_target, return_counts=True)[1]
        print("Class sample counts", class_sample_count)
        class_sample_count[0] *= 3
        class_sample_count[1] *= 2
        class_sample_count[2] *= 2.3
        class_sample_count[4] *= 2.7
        print("After class sample counts", class_sample_count)
        weight = 1. / class_sample_count
        samples_weight = weight[train_target]  # unpacks
        samples_weight = torch.from_numpy(samples_weight)
        self.sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler,
                          pin_memory=True, shuffle=False, num_workers=self.config.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_dataset, batch_size=self.config.valid_bs,
                          pin_memory=True, shuffle=False, num_workers=self.config.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_loader, batch_size=self.config.valid_bs,
                          pin_memory=True, shuffle=False, num_workers=self.config.num_workers)
