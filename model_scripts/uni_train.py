import os

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

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
    # if you want to load a checkpont, set this to the path
    checkpoint_path=None,
    start_epoch=0
):

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path) 

    train_n = loader.n
    val_n = val_loader.n
    device = loader.device
    print("here")


    model = CLAM_SB(gate=False, n_classes=2, dropout=dropout, embed_dim=1536).to(device)
    print("found model")

    # if i have a checkpoint, load it instead
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Starting from epoch {start_epoch}")


    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lmbda_l2)

    losses = []
    val_losses = []
    aucs = [] 
    val_aucs = []


    torch.save(model.state_dict(), f"{model_save_path}/model_epoch_{start_epoch}.pth")
    for epoch in range(start_epoch, start_epoch + num_epochs):

        print(f'Epoch {epoch}:')
        
        model.train()

        epoch_loss = 0
        epoch_probs = []
        epoch_labels = []
        epoch_preds = []

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
            epoch_preds.append(1 if Y_prob[0, 1].detach().cpu().item() > 0.5 else 0)

            # clear memory
            torch.cuda.empty_cache()
        
        # there should hopefully be no rounding erros with this as the dataset size is small
        losses.append(epoch_loss)
        epoch_auc = auc(epoch_labels, epoch_probs)
        aucs.append(epoch_auc)

        # Calculate confusion matrix and metrics
        tn, fp, fn, tp = confusion_matrix(epoch_labels, epoch_preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate / Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        accuracy = accuracy_score(epoch_labels, epoch_preds)
        precision = precision_score(epoch_labels, epoch_preds, zero_division=0)

        print(f'Train Loss: {epoch_loss}')
        print(f'Train AUC: {epoch_auc}')
        print(f'Train Metrics - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
        print(f'Train Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}')
        
        # Log training metrics
        with open(f"{model_save_path}/training_log.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch}, Train Loss: {epoch_loss}, Train AUC: {epoch_auc}\n")
            log_file.write(f"Epoch {epoch}, Train TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
            log_file.write(f"Epoch {epoch}, Train Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}\n")

        # #### evaluating on validation data ###

        model.eval()
        
        val_loss = 0
        val_probs = []
        val_labels = []
        val_preds = []

        for features, label in tqdm(val_loader, desc=f"Val Epoch {epoch}", total=val_loader.n):

            # data shape: batch_size x embed_dim
            logits, Y_prob, _, _, _ = model.forward(features, label, instance_eval=True)
            classification_loss = loss_func(logits.squeeze(0), label)

            val_loss += classification_loss.detach().cpu().item() / val_n
            val_probs.append(Y_prob[0, 1].detach().cpu().item())
            val_labels.append(label.detach().cpu().item())
            val_preds.append(1 if Y_prob[0, 1].detach().cpu().item() > 0.5 else 0)

            # clear memory
            torch.cuda.empty_cache()
        
        val_losses.append(val_loss)
        val_auc = auc(val_labels, val_probs)
        val_aucs.append(val_auc)

        # Calculate validation confusion matrix and metrics
        val_tn, val_fp, val_fn, val_tp = confusion_matrix(val_labels, val_preds, labels=[0, 1]).ravel()
        val_sensitivity = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        val_specificity = val_tn / (val_tn + val_fp) if (val_tn + val_fp) > 0 else 0
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)

        print(f'Validation Loss: {val_loss}')
        print(f'Validation AUC: {val_auc}')
        print(f'Validation Metrics - TP: {val_tp}, FP: {val_fp}, TN: {val_tn}, FN: {val_fn}')
        print(f'Validation Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}')
        
        # Log validation metrics
        with open(f"{model_save_path}/training_log.txt", "a") as log_file:
            log_file.write(f"Epoch {epoch}, Val Loss: {val_loss}, Val AUC: {val_auc}\n")
            log_file.write(f"Epoch {epoch}, Val TP: {val_tp}, FP: {val_fp}, TN: {val_tn}, FN: {val_fn}\n")
            log_file.write(f"Epoch {epoch}, Val Sensitivity: {val_sensitivity:.4f}, Specificity: {val_specificity:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}\n")

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