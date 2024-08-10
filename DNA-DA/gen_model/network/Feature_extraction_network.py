import torch.nn as nn


class CNN_Feature_Extraction_Network(nn.Module):
    def __init__(self, conv1_in_channels, conv1_out_channels, conv2_out_channels, conv_kernel_size_num, pool_kernel_size_num, device):
        super(CNN_Feature_Extraction_Network, self).__init__()

        self.device = device
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_in_channels, out_channels=conv1_out_channels,
                      kernel_size=(1, conv_kernel_size_num)),
            nn.BatchNorm2d(conv1_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pool_kernel_size_num), stride=pool_kernel_size_num)
        ).to(self.device)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=conv1_out_channels, out_channels=conv2_out_channels,
                      kernel_size=(1, conv_kernel_size_num)),
            nn.BatchNorm2d(conv2_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, pool_kernel_size_num), stride=pool_kernel_size_num)
        ).to(self.device)

    def forward(self, x):
        raw_feature_x = self.conv2(self.conv1(x))
        one_dimension_x = raw_feature_x.view(raw_feature_x.size(0), raw_feature_x.size(1) * raw_feature_x.size(2) * raw_feature_x.size(3))
        return raw_feature_x, one_dimension_x
