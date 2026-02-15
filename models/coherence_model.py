import torch.nn as nn

class CoherenceModel(nn.Module):
    def __init__(self, embedding_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)