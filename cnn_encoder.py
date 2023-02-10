
import torch.nn as nn
from torch.nn.utils import weight_norm


class CNN_Encoder(nn.Module):
    def __init__(self, n_inputs, n_outputs):

        super(CNN_Encoder, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=1, stride=1, padding=0))
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size=1, stride=1, padding=0))
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(
                        self.conv1, 
                        self.bn1, 
                        self.relu1, 
                        self.conv2, 
                        self.bn2, 
                        self.relu2
                    )

    def forward(self, x):
        encode_x = self.net(x.transpose(1,2)).transpose(1,2)
        return encode_x