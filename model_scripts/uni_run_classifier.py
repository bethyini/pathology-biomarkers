import sys
import os
sys.path.append("/orcd/data/edboyden/002/ezh/uni")

from uni_dataloader import train_val_test_split
from uni_train import train

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True, help="Biomarker label name (e.g., mmr_status)")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout value")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
args = parser.parse_args()

name = args.name

train_loader, val_loader = train_val_test_split(
    dir_name='/orcd/data/edboyden/002/ezh/uni/CLAM/TCGA-COAD_h5',
    label_name=name,
    test_size=0.0,
    random_seed=0
)

train(
    loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    lr=args.lr,
    model_save_path=f'/orcd/data/edboyden/002/ezh/uni/UNI/{name}/dropout{args.dropout}_lr{args.lr}',
    model_save_freq=1,
    dropout=args.dropout,
    name=name,
    # start_epoch=35,
    # checkpoint_path=f'/orcd/data/edboyden/002/ezh/uni/UNI/her2_log_top_25/dropout0.5_lr1e-05/model_epoch_35.pth'
)