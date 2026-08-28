import torch.nn as nn
import torch
from torch.utils.data import Dataset


class SentenceSplitDataset(Dataset):
    def __init__(self, hidden_states, labels):
        self.hidden_states = hidden_states
        self.labels = labels

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return {
            'hidden_states': self.hidden_states[idx],
            'labels': self.labels[idx]
        }


class SignalDataset(Dataset):
    def __init__(self, hidden_states, labels):
        self.hidden_states = hidden_states
        self.labels = labels

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return {
            'hidden_states': self.hidden_states[idx],
            'labels': self.labels[idx]
        }


class SentenceSplitMLP(nn.Module):
    """Predicts sentence boundary indicators from hidden states."""
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1, dtype=torch.bfloat16),
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states).squeeze(-1)


class SignalPredictorMLP(nn.Module):
    """Predicts the switching signal strength from hidden states."""
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1, dtype=torch.bfloat16),
            nn.Sigmoid()
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states).squeeze(-1)
