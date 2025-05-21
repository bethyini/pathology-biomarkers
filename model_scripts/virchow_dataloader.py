import os
import random
import sys
sys.path.append("/orcd/data/edboyden/002/ezh/uni")

import h5py

from sklearn.model_selection import train_test_split

import numpy as np
import torch

def train_val_test_split(
    train_size = 0.8,
    val_size = 0.2,
    test_size = 0.0,
    label_name = 'mmr_status',
    dir_name = 'h5_features',
    random_seed = None
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
        f.close()

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
    train_loader = data_loader(train_files, label_name, train_labels)
    val_loader = data_loader(val_files, label_name, weighted_sample=False)

    if test_size > 0.0:
        test_loader = data_loader(test_files, label_name, weighted_sample=False)
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader


class data_loader():

    def __init__(
        self,
        file_paths,
        label_name = 'mmr_status',
        labels = None,
        weighted_sample = True,
        device = 'cuda'
    ):
        self.file_paths = file_paths
        self.n = len(file_paths)
        self.label_name = label_name
        self.device = torch.device(device)
        
        # variables for iteration
        self.index = 0

        self.weighted_sample = weighted_sample


        if self.weighted_sample:

            if labels is None:
                labels = np.zeros(n)
                for i in range(self.n):
                    file_path = self.file_paths[i]
                    f = h5py.File(file_path, 'r')
                    labels[i] = f[self.label_name]
                    f.close()

            count_1 = labels.sum()
            count_0 = self.n - count_1
            weights = np.where(labels == 1, 1/count_1, 1/count_0)
            self.weights = weights / np.sum(weights)

        else:

            self.weights = None




    def __iter__(self):
        self.index = 0
        return self

    def __next__(
        self
    ):
        if self.index >= self.n:
            raise StopIteration

        if self.weights is None: idx = self.index
        else: idx = np.random.choice(self.n, p=self.weights)
        self.index += 1

        file_path = self.file_paths[idx]
        f = h5py.File(file_path, 'r')
        label = torch.tensor(np.array(f[self.label_name]), dtype=torch.int64)

        ### MODIFICIATION ######################################################
        # to accomodate for both virchow and uni

        features = np.array(f['features'])
        if features.ndim == 3:
            features = torch.tensor(features, dtype=torch.float32).squeeze(0)
        else:
            features = torch.tensor(features, dtype=torch.float32)

        ########################################################################

        f.close()
        return features.to(self.device), label.to(self.device)
