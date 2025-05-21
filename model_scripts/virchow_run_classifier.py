import sys
import os
import h5py
import numpy as np

sys.path.append("/orcd/data/edboyden/002/ezh/uni")

from virchow_dataloader import train_val_test_split
from virchow_train import train

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="Biomarker label name (e.g., mmr_status)")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout value")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
args = parser.parse_args()

name = args.name

train_loader, val_loader = train_val_test_split(
    dir_name='/orcd/data/edboyden/002/ezh/uni/virchow_features',
    label_name=name,
    test_size=0.0,
    random_seed=0
)


# ADD THESE CHECKS HERE:
print("\n=== DATA VERIFICATION CHECKS ===")

# Check 1: Label distribution
train_labels = []
val_labels = []

# Extract labels from loaders
for i in range(train_loader.n):
    f = h5py.File(train_loader.file_paths[i], 'r')
    train_labels.append(f[name][()])
    f.close()

for i in range(val_loader.n):
    f = h5py.File(val_loader.file_paths[i], 'r')
    val_labels.append(f[name][()])
    f.close()

print(f"Train label distribution: {np.bincount(train_labels)}")
print(f"Val label distribution: {np.bincount(val_labels)}")
print(f"Train label mean: {np.mean(train_labels):.3f}")
print(f"Val label mean: {np.mean(val_labels):.3f}")

# Check 2: Verify no overlap
train_set = set(train_loader.file_paths)
val_set = set(val_loader.file_paths)
print(f"Train/Val overlap: {len(train_set.intersection(val_set))} files")

# Check 3: Sample verification
print("\nSample verification:")
for i in range(min(3, train_loader.n)):
    f = h5py.File(train_loader.file_paths[i], 'r')
    print(f"Train file {i}: {os.path.basename(train_loader.file_paths[i])}, Label: {f[name][()]}")
    f.close()

# Check 4: Feature statistics
print("\nFeature statistics (first 5 samples):")
for i in range(min(5, train_loader.n)):
    f = h5py.File(train_loader.file_paths[i], 'r')
    features = f['features'][()]
    print(f"File {i}: shape={features.shape}, mean={features.mean():.3f}, std={features.std():.3f}")
    f.close()

print("=== END VERIFICATION ===\n")



train(
    loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    lr=args.lr,
    model_save_path=f'/orcd/data/edboyden/002/ezh/uni/virchow/{name}/dropout{args.dropout}_lr{args.lr}',
    model_save_freq=1,
    dropout=args.dropout,
    name=name,
    # start_epoch=60,
    # checkpoint_path=f'/orcd/data/edboyden/002/ezh/uni/virchow/{name}/dropout{args.dropout}_lr{args.lr}/model_epoch_60.pth'
)