import math, torch, torch.nn as nn, torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, d=512, p=0.1):
        super().__init__()
        self.drop = nn.Dropout(p)
        self.lin = nn.Linear(d, d)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x):
        return F.normalize(x + self.lin(self.drop(x)), dim=1)

class ArcMargin(nn.Module):
    def __init__(self, d, n_classes, s=30.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, d) * 0.01)
        self.s, self.m = s, m

    def forward(self, z, y=None):
        cos = F.linear(F.normalize(z), F.normalize(self.W)).clamp(-1+1e-7, 1-1e-7)
        if y is None:
            return cos
        target = torch.cos(torch.acos(cos) + self.m)
        onehot = F.one_hot(y, cos.size(1)).bool()
        return torch.where(onehot, target, cos) * self.s