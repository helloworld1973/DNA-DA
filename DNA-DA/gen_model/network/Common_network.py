import torch.nn as nn


class linear_classifier(nn.Module):
    def __init__(self, input_dim, class_num, device):
        super(linear_classifier, self).__init__()
        self.device = device
        self.fc = nn.Linear(input_dim, class_num).to(self.device)

    def forward(self, x):
        x = self.fc(x)
        return x
