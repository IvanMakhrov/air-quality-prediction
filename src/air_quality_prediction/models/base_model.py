import torch
from torch import nn


class NeuralNet(nn.Module):
    """
    Simple feedforward neural network for AQI regression.

    Architecture:
        Linear(input_size → hidden_size) → ReLU → Dropout → Linear(hidden_size → 1)
    """

    def __init__(self, input_size: int, hidden_size: int = 128, dropout_rate: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_size)
        Returns:
            (batch_size,)
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1)
