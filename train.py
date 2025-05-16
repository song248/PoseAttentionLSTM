import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from models.pa_lstm import PoseAttentionLSTM


# ✅ Dataset 클래스
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
        X = torch.tensor(self.X_seq[idx], dtype=torch.float32)
        y = torch.tensor(self.y_seq[idx], dtype=torch.float32)
        return X, y


# ✅ Train
def train(model, dataloader, optimizer, criterion, device, grad_clip=None):
    model.train()
    losses, preds, targets = [], [], []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
        preds.extend((output > 0.5).cpu().numpy())
        targets.extend(y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return np.mean(losses), acc


# ✅ Eval
def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            losses.append(loss.item())
            preds.extend((output > 0.5).cpu().numpy())
            targets.extend(y.cpu().numpy())
    acc = accuracy_score(targets, preds)
    return np.mean(losses), acc


# ✅ Main 함수
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
        num_joints=dataset.X_seq.shape[2],
        input_dim=dataset.X_seq.shape[3] if dataset.X_seq.ndim == 4 else 2,
        hidden_dim=128,
        num_layers=1,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, grad_clip=grad_clip)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

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


# ✅ 실행 진입점
if __name__ == "__main__":
    features_path = "features.npy"
    labels_path = "labels.npy"
    batch_size = 64
    epochs = 100
    lr = 1e-4
    seq_len = 30
    grad_clip = 1.0
    patience = 10
    model_path = "model/fight/pa_lstm_fight_model.pth"

    main(features_path, labels_path, batch_size, epochs, lr, seq_len, grad_clip, patience, model_path)
