import os
import random

import h5py

from sklearn.model_selection import train_test_split

import numpy as np
import torch

def train_val_test_split(
    train_size = 0.7,
    val_size = 0.3,
    test_size = 0.0,
    label_name=None,
    dir_name='CLAM/TCGA-COAD_h5',
    random_seed=0
):

    file_names = os.listdir(dir_name)
    filtered_files = []
    labels = []

    # remove non .h5 files and files without desired label
    for file_name in file_names:
        if file_name.split('.')[-1] != 'h5': continue
        file_path = f'{dir_name}/{file_name}'
        f = h5py.File(file_path, 'r')
        if label_name in f.keys():
            filtered_files.append(file_path)
            labels.append(np.array(f[label_name]).item())

    assert len(filtered_files) != 0, "No Files Found. Check input 'label_name'."

    filtered_files = np.array(filtered_files)
    labels = np.array(labels)

    # split in train, validation, and test sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        filtered_files, labels, train_size=train_size, stratify=labels, 
        random_state=random_seed)
    print(f'Training: {len(train_files)} images,', 
        f'{train_labels.sum()} Positive Class ({train_labels.mean() * 100:.2f}%)')

    if test_size > 0.0:
        val_files, test_files, val_labels, test_labels = train_test_split(
            val_files, val_labels, train_size=val_size/(val_size + test_size), 
            stratify=val_labels, random_state=random_seed)
        
    print(f'Validation: {len(val_files)} images,', 
    f'{val_labels.sum()} Positive Class ({val_labels.mean() * 100:.2f}%)')
    
    if test_size > 0.0:
        print(f'Test: {len(test_files)} images,', 
            f'{test_labels.sum()} Positive Class ({test_labels.mean() * 100:.2f}%)')

    # create dataloaders
    train_loader = data_loader(train_files, label_name)
    val_loader = data_loader(val_files, label_name)

    if test_size > 0.0:
        test_loader = data_loader(test_files, label_name)
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader


class data_loader():

    def __init__(
        self,
        file_paths,
        label_name,
        device = 'cuda'
    ):
        self.file_paths = file_paths
        self.n = len(file_paths)
        self.label_name = label_name
        self.device = device  # Add this line
        self.cuda = torch.device(device)
        
        # variables for iteration
        self.index = 0
        self.index_order = np.arange(0, self.n, 1)
        np.random.shuffle(self.index_order)


    def __iter__(self):
        np.random.shuffle(self.index_order)
        self.index = 0
        return self

    def __next__(
        self
    ):
        if self.index >= self.n:
            raise StopIteration

        idx = self.index_order[self.index]
        self.index += 1

        file_path = self.file_paths[idx]
        f = h5py.File(file_path, 'r')
        label = torch.tensor(np.array(f[self.label_name]))
        features = torch.tensor(np.array(f['features'])).squeeze(0)
        return features.to(self.device), label.to(self.device)
