import sys

import torch
from pandas import DataFrame
from torch import Tensor
from torch.optim import SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from cassava_dataset import CassavaDataset
from common_utils import get_train_transforms, get_valid_transforms, get_data_dfs_from_fold
from config import Configuration

sys.path.append('../pytorch-image-models')
import timm  # pytorch-image-models implementations

from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.metrics import Accuracy
from typing import Optional, Union, List, Any, Dict, Callable

import numpy as np


# EfficientNet noisy student: https://arxiv.org/pdf/1911.04252.pdf.
# Implementation from https://github.com/rwightman/pytorch-image-models.
class LightningModel(LightningModule):
    def __init__(self, config: Configuration, criterion, lr=0.1,
                 fc_nodes=0, pretrained=False, bn=True, features=False):
        super().__init__()
        self.valid_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.test_predictions = []
        self.config = config
        self.criterion = criterion
        self.lr = lr
        self.model = timm.create_model(config.model_arch, pretrained=pretrained)
        # replace classifier with a Linear in_features->n_classes layer
        in_features = self.model.classifier.in_features
        n_classes = self.config.num_classes
        if fc_nodes:
            self.model.classifier = torch.nn.Linear(in_features, fc_nodes)
            self.model.fc2 = torch.nn.Linear(fc_nodes, n_classes)
        else:
            self.model.classifier = torch.nn.Linear(in_features, n_classes)

        self.freeze_layers(bn, features)

    def freeze_layers(self, bn, features):
        for name, param in self.named_parameters():
            if 'model.classifier' not in name:
                param.requires_grad = False if features else True

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
                for p in module.parameters():
                    p.requires_grad = False if bn else True

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        print('opt lr', self.lr)
        opt = SGD(self.parameters(), lr=self.lr, momentum=self.config.momentum)

        # opt = Adam(self.model.parameters(), self.config.lr,
        # weight_decay=self.config.weight_decay, amsgrad=self.config.is_amsgrad)
        # opt = AdaBound(self.model.parameters(), self.config.lr)

        scheduler = CosineAnnealingWarmRestarts(opt, T_0=self.config.T_0,
                                                T_mult=self.config.T_mult,
                                                eta_min=self.config.min_lr)

        return [opt], [scheduler]

    def optimizer_step(self, epoch: int = None, batch_idx: int = None, optimizer: Optimizer = None,
                       optimizer_idx: int = None, optimizer_closure: Optional[Callable] = None, on_tpu: bool = None,
                       using_native_amp: bool = None, using_lbfgs: bool = None) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp,
                               using_lbfgs)

    def training_step(self, batch, batch_idx):
        labels, loss, predictions = self.label_forward_pass(batch)
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
        return {'preds': predictions, 'target': labels}

    def test_step_end(self, outputs):
        most_likely = torch.argmax(outputs['preds'], dim=1)  # highest preds for each sample in batch
        self.test_predictions.extend(most_likely.cpu().numpy())
        self.test_accuracy(most_likely, outputs['target'])

    def on_test_epoch_end(self) -> None:
        self.log('test_acc', self.test_accuracy.compute(), prog_bar=True, logger=True)


class LightningData(LightningDataModule):

    def __init__(self, holdout_df: DataFrame, config: Configuration, folds_df: DataFrame = None, kaggle=False):
        super().__init__()
        self.folds_df = folds_df
        self.holdout_df = holdout_df
        self.config = config
        self.batch_size = config.train_bs
        self.fold_range = [str(i) for i in range(self.config.fold_num)]
        self.train_transforms = get_train_transforms(self.config.img_size)
        self.valid_transforms = get_valid_transforms(self.config.img_size)
        self.kaggle = kaggle

    def setup(self, stage: Optional[str] = None):
        if stage == 'holdout' or stage == 'ensemble_holdout' or stage == 'ensemble_test':
            self.test_loader = CassavaDataset(self.holdout_df, self.config.train_img_dir,
                                              output_label=(not self.kaggle),
                                              transform=self.valid_transforms)
        elif stage in self.fold_range:
            train_df, valid_df = get_data_dfs_from_fold(self.folds_df, int(stage))
            self.train_dataset = CassavaDataset(train_df, self.config.train_img_dir, output_label=True,
                                                transform=self.train_transforms)
            self.valid_dataset = CassavaDataset(valid_df, self.config.train_img_dir, output_label=True,
                                                transform=self.valid_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          pin_memory=True, shuffle=True, num_workers=self.config.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_dataset, batch_size=self.config.valid_bs,
                          pin_memory=True, shuffle=False, num_workers=self.config.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_loader, batch_size=self.config.valid_bs,
                          pin_memory=True, shuffle=False, num_workers=self.config.num_workers)
