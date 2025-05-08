import os

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score as auc

import torch
import torch.nn as nn

from models.virchow_clam import CLAM_SB

def train(
    loader,
    val_loader,
    name=None,
    # training hyperparameters
    num_epochs = 10,
    lr=1e-5,
    lmbda_l2=1e-3,
    bag_weight = 0.7, # weighting of classification loss (against instance loss)
    # model saving 
    model_save_path='model',
    model_save_freq=1, # how often to save model weights
    validate=True, 
    # model arguments
    gate=True,
    dropout=0.25,

    ### ADDED PARAMETERS #######################################################
    input_dim=2560, #2560, # for virchow; 1536 for uni
    hidden_dim=512, # CLAM default 
    attn_net_dim=256 # CLAM default for size_arg="small", 384 for big
    ############################################################################

):

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path) 

    train_n = loader.n
    val_n = val_loader.n
    device = loader.device


    model = CLAM_SB(gate=gate, n_classes=2, dropout=dropout, embed_dim=input_dim,
        hidden_dim=hidden_dim, attn_net_dim=attn_net_dim).to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lmbda_l2)

    losses = []
    val_losses = []

    if validate:
        aucs = [] 
        val_aucs = []


    torch.save(model.state_dict(), f"{model_save_path}/model_epoch_{0}.pth")

    for epoch in range(num_epochs):

        print(f'Epoch {epoch + 1}:')
        
        model.train()

        epoch_loss = 0
        epoch_probs = []
        epoch_labels = []

        for features, label in loader:

            optimizer.zero_grad()
            # data shape: batch_size x embed_dim
            logits, Y_prob, _, _, results_dict = model.forward(features, label, instance_eval=True)
            instance_loss = results_dict['instance_loss']
            classification_loss = loss_func(logits.squeeze(0), label)
            loss = (1 - bag_weight) * instance_loss + bag_weight * classification_loss
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

        if validate:

            model.eval()
            
            val_loss = 0
            val_probs = []
            val_labels = []

            for features, label in val_loader:

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
            if validate:
                plt.plot(val_losses, label='Validation Loss')
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.title(f"{name} Classification Loss Across Epochs, Dropout = {dropout}")
            plt.legend()
            plt.savefig(f"{model_save_path}/loss.png")
            plt.clf()

            plt.plot(aucs, label='Training AUC')
            if validate:
                plt.plot(val_aucs, label='Validation AUC')
            plt.xlabel("epochs")
            plt.ylabel("AUC")
            plt.title(f"{name} AUC across Epochs, Dropout={dropout}")
            plt.legend()
            plt.ylim(0.5, 1.0)
            plt.savefig(f"{model_save_path}/aucs.png")
            plt.clf()