import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from model.pa_lstm import PoseAttentionLSTM
from train import PoseSequenceDataset
from torch.utils.data import DataLoader

SEQ_LEN = 10
BATCH_SIZE = 64
MODEL_PATH = "model/fight/pa_lstm_fight_model.pth"
# MODEL_PATH = "model/fight/(old)pa_lstm_fight_model.pth"
# MODEL_PATH = "model/fight/(100epoch)pa_lstm_fight_model.pth"
# MODEL_PATH = "model/fight/(200epoch)pa_lstm_fight_model.pth"
FEATURE_PATH = "test_features.npy"
LABEL_PATH = "test_labels.npy"
OUTPUT_DIR = "results"
THRESHOLD = 0.5 

os.makedirs(OUTPUT_DIR, exist_ok=True)
dataset = PoseSequenceDataset(FEATURE_PATH, LABEL_PATH, seq_len=SEQ_LEN)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseAttentionLSTM(
    num_joints=8,
    input_dim=2,
    hidden_dim=128,
    num_layers=1,
    dropout=0.3
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

y_true = []
y_pred = []
y_score = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device).squeeze()
        outputs = model(X)
        y_true.extend(y.cpu().numpy())
        y_score.extend(outputs.cpu().numpy())
        y_pred.extend((outputs > THRESHOLD).float().cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=["Normal", "Violence"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "Violence"],
                yticklabels=["Normal", "Violence"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✅] Saved confusion matrix to {save_path}")
    plt.close()

def plot_roc_curve(y_true, y_score, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✅] Saved ROC curve to {save_path}")
    plt.close()

plot_confusion_matrix(y_true, y_pred, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plot_roc_curve(y_true, y_score, os.path.join(OUTPUT_DIR, "roc_curve.png"))

