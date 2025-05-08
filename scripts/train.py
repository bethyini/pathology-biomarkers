import os

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score as auc

import torch
import torch.nn as nn

from models.clam import CLAM_SB

def train(
    loader,
    val_loader,
    name=None,
    # training hyperparameters
    num_epochs=10,
    lr=1e-5,
    lmbda_l2=1e-3,
    # model saving 
    model_save_path=None,
    model_save_freq=1,
    dropout=0.1,
):

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path) 

    train_n = loader.n
    val_n = val_loader.n
    device = loader.device
    print("here")


    model = CLAM_SB(gate=False, n_classes=2, dropout=dropout, embed_dim=1536).to(device)
    print("found model")

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lmbda_l2)

    losses = []
    val_losses = []
    aucs = [] 
    val_aucs = []


    torch.save(model.state_dict(), f"{model_save_path}/model_epoch_{0}.pth")

    for epoch in range(num_epochs):

        print(f'Epoch {epoch}:')
        
        model.train()

        epoch_loss = 0
        epoch_probs = []
        epoch_labels = []

        for features, label in tqdm(loader, desc=f"Train Epoch {epoch}", total=loader.n):
            optimizer.zero_grad()
            # data shape: batch_size x embed_dim
            logits, Y_prob, _, _, results_dict = model.forward(features, label, instance_eval=True)
            instance_loss = results_dict['instance_loss']
            classification_loss = loss_func(logits.squeeze(0), label)
            loss = instance_loss + classification_loss
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradient
            loss.backward()
            optimizer.step()

            epoch_loss += classification_loss.detach().cpu().item() / train_n
            epoch_probs.append(Y_prob[0, 1].detach().cpu().item())
            epoch_labels.append(label.detach().cpu().item())

            # clear memory
            torch.cuda.empty_cache()
        
        # there should hopefully be no rounding erros with this as the dataset size is small
        losses.append(epoch_loss)
        epoch_auc = auc(epoch_labels, epoch_probs)
        aucs.append(epoch_auc)

        print(f'Train Loss: {epoch_loss}')
        print(f'Train AUC: {epoch_auc}')

        # #### evaluating on validation data ###

        model.eval()
        
        val_loss = 0
        val_probs = []
        val_labels = []

        for features, label in tqdm(val_loader, desc=f"Val Epoch {epoch}", total=val_loader.n):

            # data shape: batch_size x embed_dim
            logits, Y_prob, _, _, _ = model.forward(features, label, instance_eval=True)
            classification_loss = loss_func(logits.squeeze(0), label)

            val_loss += classification_loss.detach().cpu().item() / val_n
            val_probs.append(Y_prob[0, 1].detach().cpu().item())
            val_labels.append(label.detach().cpu().item())

            # clear memory
            torch.cuda.empty_cache()
        
        val_losses.append(val_loss)
        val_auc = auc(val_labels, val_probs)
        val_aucs.append(val_auc)

        print(f'Validation Loss: {val_loss}')
        print(f'Validation AUC: {val_auc}')

        # saving models and plots

        if (epoch+1) % model_save_freq == 0:
            torch.save(model.state_dict(), f"{model_save_path}/model_epoch_{epoch+1}.pth")

            plt.plot(losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title(f"{name} Classification Loss Across Epochs, Dropout = {dropout}")
            plt.legend()
            plt.savefig(f"{model_save_path}/loss.png")
            plt.clf()

            plt.plot(aucs, label='Training AUC')
            plt.plot(val_aucs, label='Validation AUC')
            plt.xlabel("Epochs")
            plt.ylabel("AUC")
            plt.title(f"{name} AUC across Epochs, Dropout={dropout}")
            plt.legend()
            plt.savefig(f"{model_save_path}/aucs.png")
            plt.clf()