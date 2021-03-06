import os
import random
from collections import OrderedDict, defaultdict

import numpy as np
from sklearn.model_selection import StratifiedKFold
from albumentations import (
    HorizontalFlip, VerticalFlip, Transpose, HueSaturationValue, RandomResizedCrop,
    RandomBrightnessContrast, Compose, Normalize, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
import torch
from sklearn.model_selection import train_test_split


def set_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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


def make_holdout_df(train_df, holdout_proportion=0.15, seed=None):
    train_df, holdout_df = train_test_split(train_df, test_size=holdout_proportion, random_state=seed)
    train_df = train_df.reset_index(drop=True)
    holdout_df = holdout_df.reset_index(drop=True)
    return train_df, holdout_df


def get_train_transforms(image_size):
    return Compose([
        RandomResizedCrop(image_size, image_size),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(p=0.5),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        CoarseDropout(p=0.5),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms(image_size):
    return Compose([
        CenterCrop(image_size, image_size, p=1.),
        Resize(image_size, image_size),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_tta_transforms(image_size):
    return Compose([
        CenterCrop(image_size, image_size, p=1.),
        Resize(image_size, image_size),
        HorizontalFlip(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def average_model_state(checkpoint_fnames):
    avg_state_dict = {}
    avg_count = {}
    for ckpt in checkpoint_fnames:
        state_dict = torch.load(ckpt)['state_dict']
        for k, v in state_dict.items():
            cloned_v = v.clone().to(dtype=torch.float64)
            if k not in avg_state_dict:
                avg_state_dict[k] = cloned_v
                avg_count[k] = 1
            else:
                avg_state_dict[k] += cloned_v
                avg_count[k] += 1

    for k, v in avg_state_dict.items():
        v.div_(avg_count[k])

    # float32 overflow seems unlikely based on weights seen to date, but who knows
    float32_info = torch.finfo(torch.float32)
    final_state_dict = {}
    for k, v in avg_state_dict.items():
        v = v.clamp(float32_info.min, float32_info.max)
        final_state_dict[k] = v.to(dtype=torch.float32)

    return final_state_dict
