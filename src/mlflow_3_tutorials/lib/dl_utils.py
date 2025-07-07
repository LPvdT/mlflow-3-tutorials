import pandas as pd
import torch
from torch import nn


def prepare_data(df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    """Helper function to prepare data"""

    X = torch.tensor(df.iloc[:, :-1].values, dtype=torch.float32)
    y = torch.tensor(df.iloc[:, -1].values, dtype=torch.long)

    return X, y


def compute_accuracy(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor
) -> float:
    """Helper function to compute accuracy"""

    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        return (predicted == y).sum().item() / y.size(0)


class IrisClassifier(nn.Module):
    """Basic model for Iris classification"""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)

        return self.fc2(x)
