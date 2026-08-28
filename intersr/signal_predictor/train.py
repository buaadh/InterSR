import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pickle
from intersr.signal_predictor.architecture import SentenceSplitDataset, SignalDataset, SentenceSplitMLP, SignalPredictorMLP
from tqdm import tqdm
import numpy as np
import sys
from sklearn.metrics import confusion_matrix, f1_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        bce_loss = self.bce(input, target.float())
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

def evaluate(model, dataloader, loss_fn, device, is_split=True, zero_weight=0.1, print_detail=False):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    gt_list = []
    pred_list = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['hidden_states'].to(device)
            y = batch['labels'].to(device)
            if is_split:
                logits = model(x)
                loss = loss_fn(logits, y.float())
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.append(preds.cpu())
                all_labels.append(y.cpu())
            else:
                pred = model(x)
                # Weight loss for labels with value 0
                weights = torch.ones_like(y, dtype=torch.float, device=device)
                weights[y == 0] = zero_weight
                loss = ((pred - y.float()) ** 2 * weights).mean()
                gt_list.append(y.cpu())
                pred_list.append(pred.cpu())
            losses.append(loss.item())
    if is_split:
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        acc = (all_preds == all_labels).float().mean().item()
        cm = confusion_matrix(all_labels.numpy().flatten(), all_preds.numpy().flatten())
        f1 = f1_score(all_labels.numpy().flatten(), all_preds.numpy().flatten(), average='macro')
        return np.mean(losses), acc, cm, f1
    else:
        if print_detail:
            gt = torch.cat(gt_list, dim=0).view(-1).to(torch.float32).numpy()
            pred = torch.cat(pred_list, dim=0).view(-1).to(torch.float32).numpy()
            print("Signal predict first 10 items:")
            for i in range(min(10, len(gt))):
                print(f"GT: {gt[i]:.4f}\tPred: {pred[i]:.4f}")
        return np.mean(losses)

def train_sentence_split(model, train_loader, test_loader, device, output_dir, max_epochs=20, early_stop_rounds=3):
    loss_fn = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    best_f1 = 0
    not_improve_count = 0
    for epoch in range(max_epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Split] Epoch {epoch+1}", file=sys.stdout)
        for i, batch in pbar:
            x = batch['hidden_states'].to(device)
            y = batch['labels'].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y.float())
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            # Evaluate every 10 batches
            if (i+1) % 10 == 0:
                val_loss, val_acc, val_cm, val_f1 = evaluate(model, test_loader, loss_fn, device, is_split=True)
                pbar.set_postfix({'loss': loss.item(), 'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1})
                print(f"[Split] Epoch {epoch+1} Batch {i+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")
                print("Confusion Matrix:\n", val_cm)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    not_improve_count = 0
                    torch.save(model.state_dict(), os.path.join(output_dir, 'sentence_split_best.pt'))
                else:
                    not_improve_count += 1
                if not_improve_count >= early_stop_rounds:
                    print(f"Early stopping at epoch {epoch+1}, batch {i+1}")
                    return
    print("Training finished. Best val f1:", best_f1)

def train_signal_predictor(model, train_loader, test_loader, device, output_dir, zero_weight=0.1, max_epochs=20, early_stop_rounds=3):
    loss_fn = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    best_loss = float('inf')
    not_improve_count = 0
    for epoch in range(max_epochs):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[Signal] Epoch {epoch+1}", file=sys.stdout)
        for i, batch in pbar:
            x = batch['hidden_states'].to(device)
            y = batch['labels'].to(device)
            optimizer.zero_grad()
            pred = model(x)
            weights = torch.ones_like(y, dtype=torch.float, device=device)
            weights[y == 0] = zero_weight
            loss = ((pred - y.float()) ** 2 * weights).mean()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': loss.item()})
            # Evaluate every 10 batches
            if (i+1) % 10 == 0:
                val_loss = evaluate(model, test_loader, loss_fn, device, is_split=False, zero_weight=zero_weight, print_detail=True)
                pbar.set_postfix({'loss': loss.item(), 'val_loss': val_loss})
                if val_loss < best_loss:
                    best_loss = val_loss
                    not_improve_count = 0
                    torch.save(model.state_dict(), os.path.join(output_dir, 'signal_predictor_best.pt'))
                else:
                    not_improve_count += 1
                if not_improve_count >= early_stop_rounds:
                    print(f"Early stopping at epoch {epoch+1}, batch {i+1}")
                    return
    print("Training finished. Best val loss:", best_loss)

if __name__ == "__main__":
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
    output_dir = f"./output/{model_name}"
    split_batch_size = 128
    signal_batch_size = 128
    zero_weight = 0.1
    early_stop_rounds = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    with open(os.path.join(output_dir, 'sentence_split_dataset_train.pkl'), 'rb') as f:
        split_train = pickle.load(f)
    with open(os.path.join(output_dir, 'sentence_split_dataset_test.pkl'), 'rb') as f:
        split_test = pickle.load(f)
    with open(os.path.join(output_dir, 'signal_dataset_train.pkl'), 'rb') as f:
        signal_train = pickle.load(f)
    with open(os.path.join(output_dir, 'signal_dataset_test.pkl'), 'rb') as f:
        signal_test = pickle.load(f)
    # Only take first 1000 items as test
    split_test = torch.utils.data.Subset(split_test, range(min(10000, len(split_test))))
    signal_test = torch.utils.data.Subset(signal_test, range(min(10000, len(signal_test))))

    split_train_loader = DataLoader(split_train, batch_size=split_batch_size, shuffle=True)
    split_test_loader = DataLoader(split_test, batch_size=split_batch_size, shuffle=False)
    signal_train_loader = DataLoader(signal_train, batch_size=signal_batch_size, shuffle=True)
    signal_test_loader = DataLoader(signal_test, batch_size=signal_batch_size, shuffle=False)

    # Train sentence split
    split_model = SentenceSplitMLP(hidden_size=split_train.hidden_states.shape[1]).to(device)
    train_sentence_split(split_model, split_train_loader, split_test_loader, device, output_dir, max_epochs=10, early_stop_rounds=3)

    # Train signal predictor
    signal_model = SignalPredictorMLP(hidden_size=signal_train.hidden_states.shape[1]).to(device)
    train_signal_predictor(signal_model, signal_train_loader, signal_test_loader, device, output_dir, zero_weight=zero_weight, max_epochs=10, early_stop_rounds=3)
