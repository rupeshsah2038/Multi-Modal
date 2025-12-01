import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, x):
        return self.head(x)
