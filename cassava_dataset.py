from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

class CassavaDataset(Dataset):
    def __init__(self, df, data_root_dir, transform=None, output_label=False):
        self.df = df.reset_index(drop=True).copy()
        self.data_root_dir = data_root_dir
        self.transform = transform
        self.output_label = output_label
        self.labels = self.df.label.values if output_label else None
 
    def get_image(self, path):
        img_bgr = cv2.imread(path)
        img_rgb = img_bgr[:, :, ::-1]
        return img_rgb
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = '{}/{}'.format(self.data_root_dir, self.df.image_id[idx])
        img = self.get_image(path)
        
        if self.transform:
             img = self.transform(image=img)['image']
                
        if self.output_label == True:
            return img, torch.tensor(self.labels[idx].astype(np.int))
        else:
            return img