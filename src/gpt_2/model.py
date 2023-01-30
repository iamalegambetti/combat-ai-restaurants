import torch
import torch.nn as nn
from transformers import GPT2Model

class GPT2(nn.Module):
    def __init__(self, backbone, last_hidden_size = 768):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(last_hidden_size, 1)

    def forward(self, x):
        x = self.backbone(x).last_hidden_state
        x = nn.functional.relu(x)
        x = self.fc(x)
        x = torch.mean(x)
        return x