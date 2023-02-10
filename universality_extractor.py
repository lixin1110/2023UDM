import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class U_E():
    def __init__(self, timestep, dropout, hidden_size, device):
        super(U_E, self).__init__()

        self.timestep = timestep
        self.device = device
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.dropout = nn.Dropout(p = dropout)
        for i in range(self.timestep):
            if i == 0:
                tmpconv = weight_norm(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1))
            else:
                tmpconv = weight_norm(nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1))
            self.convs.append(tmpconv)
            self.bns.append(nn.BatchNorm1d(hidden_size))
        self.convs.to(device)
        self.bns.to(device)
        self.shared_gru = nn.GRU(hidden_size, 
                        hidden_size, 
                        num_layers=1, 
                        batch_first=True,
                        dropout=0.4).to(device)
    
    def forward(self, encode_samples):
        self.shared_gru.flatten_parameters()
        regressors = []
        currentR = encode_samples.transpose(1,2)
        for i in range(self.timestep):
            currentR = self.convs[i](currentR)
            currentR = self.bns[i](currentR)
            currentR = F.leaky_relu(currentR, negative_slope=0.01)
            
            regressors.append(currentR)
            currentR = self.dropout(currentR)
            if i < self.timestep - 1:
                currentR = currentR.permute(0,2,1).contiguous()
                currentR = torch.transpose(currentR,1,2)

        shared_knowledge = torch.zeros((self.timestep, encode_samples.shape[0], encode_samples.shape[2])).float().to(self.device)
        for i in range(self.timestep):
            cur_R = regressors[i].transpose(1,2).contiguous()
            output, cur_result = self.shared_gru(cur_R)
            shared_knowledge[i] = torch.squeeze(cur_result, 0)
        shared_knowledge = shared_knowledge.transpose(0,1)

        return shared_knowledge