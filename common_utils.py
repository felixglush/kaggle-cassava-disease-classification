import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.optim import SGD
from torch.utils.data import DataLoader
from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue, RandomResizedCrop,
    RandomBrightnessContrast, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import torch
from config import ConstantConfig
from cassava_dataset import CassavaDataset
from model import Model
from sklearn.model_selection import train_test_split


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Reads the train.csv file and creates a dataframe out of it, optionally separating a test set that won't be used
# in any stratified cross validation or training
def read_csv(filepath, debug, num_samples=None):
    train_df = pd.read_csv(filepath, engine='python')

    if debug or num_samples:
        n = num_samples if num_samples else 200
        train_df = train_df.sample(n=n, random_state=ConstantConfig.seed).reset_index(drop=True)

    return train_df


def make_holdout_df(train_df, holdout_proportion=0.15):
    train_df, holdout_df = train_test_split(train_df, test_size=holdout_proportion, random_state=config.ConstantConfig.seed)
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)
    return train_df, holdout_df


def stratify_split(df, splits, seed, target_col):
    train_folds = df.copy()
    stratifiedFold = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    splits = stratifiedFold.split(np.zeros(len(train_folds)), train_folds[target_col])

    # label all rows of train_folds with a particular validation set fold number they are part of 
    # (to select the row for validation when splitting on that fold)
    for fold_num, (train_idxs, val_idxs) in enumerate(splits):
        train_folds.loc[val_idxs, 'fold'] = fold_num

    train_folds['fold'] = train_folds['fold'].astype(int)
    return train_folds


def get_train_transforms(image_size):
    return Compose([
        RandomResizedCrop(image_size, image_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        Cutout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms(image_size):
    return Compose([
        CenterCrop(image_size, image_size, p=1.),
        Resize(image_size, image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def setup_scheduler_loss(optimizer, lr_test, settings):
    # -------- LOSS FUNCTION --------
    criterion = torch.nn.CrossEntropyLoss()

    # -------- SCHEDULER --------
    scheduler = None
    if lr_test:
        scheduler = StepLR(optimizer, step_size=settings.step_size_lr, gamma=settings.gamma, verbose=settings.schedule_verbosity)
        return scheduler, criterion

    elif settings.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=settings.factor, patience=settings.patience,
                                      eps=settings.eps, verbose=settings.schedule_verbosity)

    elif settings.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=settings.T_max,
                                      eta_min=settings.min_lr, verbose=settings.schedule_verbosity)

    elif settings.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=settings.T_0, T_mult=settings.T_mult,
                                                eta_min=settings.min_lr, verbose=settings.schedule_verbosity)

    return scheduler, criterion


def setup_model_optimizer(model_arch, lr, is_amsgrad, num_labels, fc_nodes, weight_decay,
                          device, checkpoint=None, nesterov=False, momentum=None):
    # -------- MODEL INSTANTIATION --------
    model = Model(model_arch, num_labels, fc_nodes, pretrained=True).to(device)

    # -------- OPTIMIZER --------
    # optimizer = Adam(model.parameters(), lr, weight_decay=weight_decay, amsgrad=is_amsgrad)
    optimizer = SGD(model.parameters(), lr, momentum=momentum, nesterov=nesterov)
    # optimizer = AdaBound(model.parameters(), lr)

    if checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    return model, optimizer


# df: stratified
def get_data_dfs(df, fold):
    train_idx = df[df['fold'] != fold].index
    valid_idx = df[df['fold'] == fold].index
    train_df = df.iloc[train_idx].reset_index(
        drop=True)  # since we are selecting rows, the index will be missing #s so reset
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df


# called for each fold
def get_loaders(train_df, valid_df, train_batchsize, data_root_dir):
    train_dataset = CassavaDataset(train_df, data_root_dir, output_label=True,
                                   transform=get_train_transforms(config.ConstantConfig.img_size))
    valid_dataset = CassavaDataset(valid_df, data_root_dir, output_label=True,
                                   transform=get_valid_transforms(config.ConstantConfig.img_size))

    train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize,
                                  pin_memory=True, shuffle=False,
                                  num_workers=Config.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=Config.valid_bs,
                                  pin_memory=True, shuffle=False,
                                  num_workers=Config.num_workers)

    return train_dataloader, valid_dataloader


# called before any folds are trained
def create_holdout_loader(df, data_root_dir):
    dataset = CassavaDataset(df, data_root_dir, output_label=True, transform=get_valid_transforms(Config.img_size))
    dataloader = DataLoader(dataset, batch_size=Config.valid_bs,
                            pin_memory=True, shuffle=False,
                            num_workers=Config.num_workers)
    targets = df.label.values
    return dataloader, targets
