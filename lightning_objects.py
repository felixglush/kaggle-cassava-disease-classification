import sys

import torch
from pandas import DataFrame
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from cassava_dataset import CassavaDataset
from common_utils import get_train_transforms, get_valid_transforms, get_data_dfs_from_fold
from config import Configuration

sys.path.append('../pytorch-image-models')
import timm  # pytorch-image-models implementations

from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.metrics import functional as FM
from typing import Optional, Union, List


# EfficientNet noisy student: https://arxiv.org/pdf/1911.04252.pdf.
# Implementation from https://github.com/rwightman/pytorch-image-models.
class LightningModel(LightningModule):
    def __init__(self, config: Configuration, criterion,
                 fc_nodes=0, pretrained=False):
        super().__init__()
        self.config = config
        self.criterion = criterion
        self.model = timm.create_model(config.model_arch, pretrained=pretrained)
        # replace classifier with a Linear in_features->n_classes layer
        in_features = self.model.classifier.in_features
        n_classes = self.config.num_classes
        if fc_nodes:
            self.model.classifier = torch.nn.Linear(in_features, fc_nodes)
            self.model.fc2 = torch.nn.Linear(fc_nodes, n_classes)
        else:
            self.model.classifier = torch.nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt = SGD(self.parameters(), self.config.lr,
                  momentum=self.config.momentum)

        # opt = Adam(self.model.parameters(), self.config.lr,
        # weight_decay=self.config.weight_decay, amsgrad=self.config.is_amsgrad)
        # opt = AdaBound(self.model.parameters(), self.config.lr)

        scheduler = CosineAnnealingWarmRestarts(opt, T_0=self.config.T_0,
                                                T_mult=self.config.T_mult,
                                                eta_min=self.config.min_lr)

        return [opt], [scheduler]

    def training_step(self, batch, batch_idx):
        _, loss, _ = self.label_forward_pass(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels, loss, predictions = self.label_forward_pass(batch)
        accuracy = FM.accuracy(torch.argmax(predictions, 1).detach().cpu().argmax(1), labels)
        metrics = {'val_acc': accuracy, 'val_loss': loss}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def label_forward_pass(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        return labels, loss, predictions

    def test_step(self, batch, batch_idx):
        pass


class LightningData(LightningDataModule):

    def __init__(self, folds_df: DataFrame, holdout_df: DataFrame, config: Configuration):
        super().__init__()
        self.folds_df = folds_df
        self.holdout_df = holdout_df
        self.config = config
        self.fold_range = [str(i) for i in range(self.config.fold_num)]
        self.train_transforms = get_train_transforms(self.config.img_size)
        self.valid_transforms = get_valid_transforms(self.config.img_size)

    def setup(self, stage: Optional[str] = None):
        if stage == 'holdout':
            self.test_loader = CassavaDataset(self.holdout_df, self.config.train_img_dir, output_label=True,
                                              transform=self.valid_transforms)
        elif stage in self.fold_range:
            train_df, valid_df = get_data_dfs_from_fold(self.folds_df, int(stage))
            self.train_dataset = CassavaDataset(train_df, self.config.train_img_dir, output_label=True,
                                                transform=self.train_transforms)
            self.valid_dataset = CassavaDataset(valid_df, self.config.train_img_dir, output_label=True,
                                                transform=self.valid_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.config.train_bs,
                          pin_memory=True, shuffle=False, num_workers=self.config.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.valid_dataset, batch_size=self.config.valid_bs,
                          pin_memory=True, shuffle=False, num_workers=self.config.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.test_loader, batch_size=self.config.valid_bs,
                          pin_memory=True, shuffle=False, num_workers=self.config.num_workers)
