import os
import sys
import numpy as np
sys.path.append("/orcd/data/edboyden/002/ezh/uni")
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
    model_save_path='/orcd/data/edboyden/002/ezh/uni/virchow',
    model_save_freq=1, # how often to save model weights
    validate=True, 
    # model arguments
    gate=True,
    dropout=0.25,

    input_dim=2560, #2560, # for virchow; 1536 for uni
    hidden_dim=512, # CLAM default 
    attn_net_dim=256, # CLAM default for size_arg="small", 384 for big

    # if you want to load a checkpont, set this to the path
    checkpoint_path=None,
    start_epoch=0

    ):


    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path) 

    log_file = open(f"{model_save_path}/training_log.txt", "a")

    train_n = loader.n
    val_n = val_loader.n
    device = loader.device

    model = CLAM_SB(gate=gate, n_classes=2, dropout=dropout, embed_dim=input_dim, hidden_dim=hidden_dim, attn_net_dim=attn_net_dim).to(device)

    # if i have a checkpoint, load it instead
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Starting from epoch {start_epoch}")


    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lmbda_l2)

    losses, aucs, val_losses, val_aucs = [], [], [], []


    torch.save(model.state_dict(), f"{model_save_path}/model_epoch_{start_epoch}.pth")


    # ADD: Debug tracking variables
    debug_interval = 5  # Print debug info every 5 epochs


    for epoch in range(start_epoch, start_epoch + num_epochs):

        print(f'Epoch {epoch + 1}:')
        
        model.train()

        epoch_loss = 0
        epoch_probs = []
        epoch_labels = []

        # ADD: Track attention weights and instance predictions
        epoch_attention_weights = []
        epoch_instance_preds = []

        for i, (features, label) in enumerate(loader):

            optimizer.zero_grad()
            # data shape: batch_size x embed_dim
            logits, Y_prob, Y_hat, A_raw, results_dict = model.forward(features, label, instance_eval=True)
            instance_loss = results_dict['instance_loss']
            classification_loss = loss_func(logits.squeeze(0), label)
            loss = (1 - bag_weight) * instance_loss + bag_weight * classification_loss
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradient
            # ADD: Debug information
            if epoch % debug_interval == 0 and i < 3:  # First 3 samples every 5 epochs
                log_file.write(f"\n  Train Sample {i}:\n")
                log_file.write(f"    True label: {label.item()}\n")
                log_file.write(f"    Predicted prob: {Y_prob[0, 1].item():.4f}\n")
                log_file.write(f"    Num patches: {features.shape[0]}\n")
                log_file.write(f"    Attention weights - mean: {A_raw.mean():.4f}, std: {A_raw.std():.4f}\n")
                log_file.write(f"    Top 5 attention weights: {A_raw.squeeze().topk(5).values.tolist()}\n")
                
                # Check instance predictions - FIX: handle numpy array
                if 'inst_preds' in results_dict:
                    inst_preds = results_dict['inst_preds']
                    if isinstance(inst_preds, np.ndarray):
                        inst_preds = torch.from_numpy(inst_preds)
                    inst_probs = torch.sigmoid(inst_preds).squeeze()
                    log_file.write(f"    Instance preds - mean: {inst_probs.mean():.4f}, std: {inst_probs.std():.4f}\n")
                    log_file.write(f"    Instance preds range: [{inst_probs.min():.4f}, {inst_probs.max():.4f}]\n")
            
                log_file.flush()

            # Store attention stats
            epoch_attention_weights.append(A_raw.mean().item())


            loss.backward()

            # ADD: Gradient monitoring
            if epoch % debug_interval == 0 and i == 0:
                total_grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"  Gradient norm: {total_grad_norm:.6f}")
                log_file.write(f"  Gradient norm: {total_grad_norm:.6f}\n")

            optimizer.step()

            epoch_loss += classification_loss.detach().cpu().item() / train_n
            epoch_probs.append(Y_prob[0, 1].detach().cpu().item())
            epoch_labels.append(label.detach().cpu().item())

            # clear memory
            torch.cuda.empty_cache()
        

        # ADD: Epoch-level statistics
        if epoch % debug_interval == 0:
            log_file.write(f"\n  Epoch {epoch} Training Statistics:\n")
            log_file.write(f"    Prediction mean: {np.mean(epoch_probs):.4f}, std: {np.std(epoch_probs):.4f}\n")
            log_file.write(f"    Prediction range: [{min(epoch_probs):.4f}, {max(epoch_probs):.4f}]\n")
            log_file.write(f"    Attention weight mean: {np.mean(epoch_attention_weights):.4f}\n")
            
            # Check prediction diversity
            unique_preds = len(set([round(p, 3) for p in epoch_probs]))
            log_file.write(f"    Unique predictions (rounded): {unique_preds}/{len(epoch_probs)}\n")

        log_file.flush()

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
            val_attention_weights = []

            for i, (features, label) in enumerate(val_loader):

                # data shape: batch_size x embed_dim
                logits, Y_prob, Y_hat, A_raw, _ = model.forward(features, label, instance_eval=False)  # instance_eval=False for validation
                classification_loss = loss_func(logits.squeeze(0), label)

                # ADD: Validation debug info
                if epoch % debug_interval == 0 and i < 3:
                    log_file.write(f"\n  Val Sample {i}:\n")
                    log_file.write(f"    True label: {label.item()}\n")
                    log_file.write(f"    Predicted prob: {Y_prob[0, 1].item():.4f}\n")
                    log_file.write(f"    Num patches: {features.shape[0]}\n")
                    log_file.write(f"    Attention weights - mean: {A_raw.mean():.4f}, std: {A_raw.std():.4f}\n")
                    log_file.flush()

                val_attention_weights.append(A_raw.mean().item())

                val_loss += classification_loss.detach().cpu().item() / val_n
                val_probs.append(Y_prob[0, 1].detach().cpu().item())
                val_labels.append(label.detach().cpu().item())

                # clear memory
                torch.cuda.empty_cache()
            
            # ADD: Validation statistics
            if epoch % debug_interval == 0:
                log_file.write(f"\n  Epoch {epoch} Validation Statistics:\n")
                log_file.write(f"    Prediction mean: {np.mean(val_probs):.4f}, std: {np.std(val_probs):.4f}\n")
                log_file.write(f"    Prediction range: [{min(val_probs):.4f}, {max(val_probs):.4f}]\n")
                log_file.write(f"    Attention weight mean: {np.mean(val_attention_weights):.4f}\n")
                
                # Check if predictions are diverse
                unique_val_preds = len(set([round(p, 3) for p in val_probs]))
                log_file.write(f"    Unique predictions (rounded): {unique_val_preds}/{len(val_probs)}\n")
                log_file.flush()


            val_losses.append(val_loss)
            val_auc = auc(val_labels, val_probs)
            val_aucs.append(val_auc)

            print(f'Validation Loss: {val_loss}')
            print(f'Validation AUC: {val_auc}')

        # saving models, plots, and logging results to file
        log_file.write(f"Epoch {epoch}:\n")
        log_file.write(f"Train Loss: {epoch_loss}\n")
        log_file.write(f"Train AUC: {epoch_auc}\n")
        if validate:
            log_file.write(f"Validation Loss: {val_loss}\n")
            log_file.write(f"Validation AUC: {val_auc}\n")
        log_file.write("\n")
        log_file.flush()

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