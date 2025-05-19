import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model.pa_lstm import PoseAttentionLSTM
from train import PoseSequenceDataset
from torch.utils.data import DataLoader

SEQ_LEN = 10
BATCH_SIZE = 64
MODEL_PATH = "model/fight/pa_lstm_fight_model.pth"

# 테스트셋 로딩
dataset = PoseSequenceDataset("test_features.npy", "test_labels.npy", seq_len=SEQ_LEN)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 준비
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

# 추론 및 결과 수집
y_true = []
y_pred = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device).squeeze()  # squeeze 추가
        outputs = model(X)
        preds = (outputs > 0.5).float()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# 결과 출력
print(classification_report(y_true, y_pred, target_names=["normal", "violence"]))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
