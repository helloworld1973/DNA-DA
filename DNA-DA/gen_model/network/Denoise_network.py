import torch
from torch import nn


class DenoiseModel(nn.Module):
    def __init__(self, n_steps, hidden_size, out_features, device):
        super(DenoiseModel, self).__init__()
        self.device = device
        self.n_steps = n_steps
        self.w = hidden_size
        self.out_features = out_features
        t_w = 1
        self.t_layer = nn.Sequential(nn.Linear(1, t_w),
                                     nn.ReLU(),
                                     nn.Linear(t_w, t_w),
                                     nn.ReLU()).to(self.device)  # Move to device here

        self.layer1 = nn.Sequential(nn.Linear(t_w + self.out_features, self.w),
                                    nn.ReLU(),
                                    nn.Linear(self.w, self.w),
                                    nn.ReLU()).to(self.device)  # Move to device here

        self.layer2 = nn.Sequential(nn.Linear(self.w, self.w),
                                    nn.ReLU(),
                                    nn.Linear(self.w, self.w),
                                    nn.ReLU()).to(self.device)  # Move to device here

        self.layer3 = nn.Sequential(nn.Linear(self.w + t_w, self.w),
                                    nn.ReLU(),
                                    nn.Linear(self.w, self.w),
                                    nn.Tanh()).to(self.device)  # Move to device here

        self.last_layer = nn.Linear(self.w, self.out_features).to(self.device)  # Move to device here

    def forward(self, x, t):
        t = (t.float() / self.n_steps) - 0.5
        temb = self.t_layer(t)

        # Concatenate tensors along the last axis
        output = torch.cat([x, temb], dim=-1)
        output = self.layer1(output)
        output = self.layer2(output)
        output = torch.cat([output, temb], dim=-1)
        output = self.layer3(output)
        return self.last_layer(output).to(self.device)  # Ensure output is on device