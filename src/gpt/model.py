import torch
import torch.nn as nn

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
    

class GPTNeo(nn.Module):
    def __init__(self, backbone, last_hidden_size = 768):
        super().__init__()
        self.backbone = backbone
        self.backbone.score = nn.Identity()
        self.fc = nn.Linear(last_hidden_size, 1)
    
    def forward(self, x):
        x = self.backbone(x).logits
        x = nn.functional.relu(x)
        x = self.fc(x)
        x = torch.mean(x)
        return x
    
if __name__ == '__main__':
    from transformers import GPT2Model, GPTNeoForSequenceClassification
    backbone = GPTNeoForSequenceClassification.from_pretrained('EleutherAI/gpt-neo-125M')
    model = GPTNeo(backbone)
    print(model)