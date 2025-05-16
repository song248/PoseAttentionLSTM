import torch
import torch.nn as nn

class PoseAttentionLSTM(nn.Module):
    def __init__(self, num_joints=17, input_dim=2, hidden_dim=128, num_layers=1, dropout=0.2):
        super(PoseAttentionLSTM, self).__init__()
        self.num_joints = num_joints
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Learnable spatial attention weights per joint
        self.attn_weights = nn.Parameter(torch.ones(num_joints), requires_grad=True)

        # LSTM to model temporal dynamics
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Fully connected classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Binary classification
            nn.Sigmoid()       # Output probability
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, J, D) where
                B = batch size
                T = sequence length (frames)
                J = number of joints
                D = input dimension (e.g., 2 or 3)
        Returns:
            pred: Tensor of shape (B,) with predicted violence probability
        """
        B, T, J, D = x.size()
        assert J == self.num_joints and D == self.input_dim, "Input shape mismatch"

        # Spatial attention (joint-wise weights)
        attn = torch.softmax(self.attn_weights, dim=0)  # Shape: (J,)
        x = x * attn[None, None, :, None]               # Broadcast to (B, T, J, D)
        x = x.sum(dim=2)                                # Sum over joints → shape: (B, T, D)

        # Temporal modeling with LSTM
        lstm_out, _ = self.lstm(x)                      # Output: (B, T, H)
        final_feat = lstm_out[:, -1, :]                 # Use last time step: (B, H)

        # Classification
        pred = self.classifier(final_feat).squeeze(-1)  # Shape: (B,)
        return pred
