from virchow_dataloader import train_val_test_split
from virchow_train import train

name = "Lymphovascular invasion indicator"
# 1. Prepare the loaders
train_loader, val_loader = train_val_test_split(
    dir_name='virchow_features',
    label_name=name,
    test_size=0.0,  # or >0 if you want test set
    random_seed=0
)

# 2. Train the model
train(
    loader=train_loader,
    val_loader=val_loader,
    num_epochs=60,
    lr=1e-5,
    model_save_path=f'virchow/{name}/dropout0.3_lr-5',
    model_save_freq=1,
    dropout=0.3,
    name=name
)
