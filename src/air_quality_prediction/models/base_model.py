import torch
from torch import nn


class BaseModel(nn.Module):
    """
    Simple feedforward neural network for AQI regression.

    Architecture:
        Linear(input_size → hidden_size) → ReLU → Dropout → Linear(hidden_size → 1)
    """

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_size)
        Returns:
            (batch_size,)
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out.squeeze(-1)
