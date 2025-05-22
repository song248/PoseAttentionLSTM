import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from model.pa_lstm import PoseAttentionLSTM

class PoseSequenceDataset(Dataset):
    def __init__(self, features_path, labels_path, seq_len=30):
        self.features = np.load(features_path)
        self.labels = np.load(labels_path)
        self.seq_len = seq_len
        self.X_seq, self.y_seq = self._build_sequences()

    def _build_sequences(self):
        X_seq, y_seq = [], []
        for i in range(len(self.features) - self.seq_len + 1):
            X_seq.append(self.features[i:i + self.seq_len])
            y_seq.append(self.labels[i + self.seq_len - 1])
        return np.array(X_seq), np.array(y_seq)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        X = self.X_seq[idx].reshape(self.seq_len, 8, 2)  # (T, 16) → (T, 8, 2)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(self.y_seq[idx], dtype=torch.float32).squeeze()
        return X, y

def train(model, dataloader, optimizer, criterion, device, grad_clip=None):
    model.train()
    losses, preds, targets = [], [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        # BCELoss
        # ------------------------------
        # output = model(X)
        # loss = criterion(output, y)
        # BCEWithLogitsLoss
        # ------------------------------
        output = model(X).unsqueeze(1) 
        output = torch.sigmoid(output)
        loss = criterion(output, y.unsqueeze(1))
        # ------------------------------
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
        preds.extend((output > 0.5).cpu().numpy())
        # preds.extend((output > 0.4).cpu().numpy())
        targets.extend(y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return np.mean(losses), acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # output = model(X)
            # loss = criterion(output, y)
            output = model(X).unsqueeze(1) 
            loss = criterion(output, y.unsqueeze(1))
            losses.append(loss.item())
            preds.extend((output > 0.5).cpu().numpy())
            targets.extend(y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return np.mean(losses), acc

def plot_training(train_losses, val_losses, train_accs, val_accs, save_path='loss_accuracy_plot.png'):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved training plot to {save_path}")

def main(features_path, labels_path, batch_size, epochs, lr, seq_len, grad_clip, patience, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = PoseSequenceDataset(features_path, labels_path, seq_len=seq_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = PoseAttentionLSTM(
        num_joints=8,
        input_dim=2,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, grad_clip=grad_clip)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    plot_training(train_losses, val_losses, train_accs, val_accs)

if __name__ == "__main__":
    features_path = "features.npy"
    labels_path = "labels.npy"
    batch_size = 64
    epochs = 200
    lr = 1e-4
    seq_len = 10
    grad_clip = 1.0
    patience = 20
    model_path = "model/fight/pa_lstm_fight_model.pth"

    main(features_path, labels_path, batch_size, epochs, lr, seq_len, grad_clip, patience, model_path)
