import torch
from torch import nn
from torch import sigmoid, exp, randn_like


class NoiseEncoder(nn.Module):
    def __init__(self, in_features, device):
        super(NoiseEncoder, self).__init__()
        self.device = device
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(in_features, int(in_features / 2)).to(device)
        self.fc2 = nn.Linear(int(in_features / 2), int(in_features / 4)).to(device)
        self.fc3_mean = nn.Linear(int(in_features / 4), in_features).to(device)
        self.fc3_logvar = nn.Linear(int(in_features / 4), in_features).to(device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        mu = self.fc3_mean(x)
        logvar = self.fc3_logvar(x)
        return mu, logvar


class NoiseDistinct_reparameterize(nn.Module):
    def __init__(self, device):
        super(NoiseDistinct_reparameterize, self).__init__()
        self.device = device

    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=self.device)
        return eps.mul(std).add_(mu)
