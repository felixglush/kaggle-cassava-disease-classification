import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from config import Config
from cassava_dataset import CassavaDataset
from model import Model

def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_csvs(data_dir, debug):
    train = pd.read_csv(data_dir + '/train.csv', engine='python')
    test = pd.read_csv(data_dir + '/sample_submission.csv', engine='python')

    if debug:
        train = train.sample(n=200, random_state=Config.seed).reset_index(drop=True)
        
    return train, test

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
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
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

def setup(model_arch, lr, is_amsgrad, num_labels, device):
     # -------- MODEL INSTANTIATION --------
    model = Model(model_arch, num_labels, pretrained=True).to(device)

    # -------- OPTIMIZER --------
    optimizer = Adam(model.parameters(), lr, weight_decay=Config.weight_decay, amsgrad=is_amsgrad)

    # -------- SCHEDULER --------
    scheduler = None
    if Config.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, factor=Config.factor, patience=Config.patience, eps=Config.eps, verbose=True)
    elif Config.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_max, eta_min=Config.min_lr, verbose=True)
    elif Config.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=Config.T_0, T_mult=Config.T_mult, eta_min=Config.min_lr, verbose=True)
    
    # -------- LOSS FUNCTION --------
    criterion = torch.nn.CrossEntropyLoss()
    
    return model, optimizer, scheduler, criterion

def get_data_dfs(df, fold):
    train_idx = df[df['fold'] != fold].index  
    valid_idx = df[df['fold'] == fold].index 
    train_df = df.iloc[train_idx].reset_index(drop=True) # since we are selecting rows, the index will be missing #s so reset
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df

def get_loaders(train_df, valid_df, train_batchsize, data_root_dir):        
    train_dataset = CassavaDataset(train_df, data_root_dir, output_label=True,
                                   transform=get_train_transforms(Config.img_size))
    valid_dataset = CassavaDataset(valid_df, data_root_dir, output_label=True, 
                                   transform=get_valid_transforms(Config.img_size))

    train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, 
                                  pin_memory=True, shuffle=False, 
                                  num_workers=Config.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=Config.valid_bs, 
                                  pin_memory=True, shuffle=False, 
                                  num_workers=Config.num_workers)
    return train_dataloader, valid_dataloader