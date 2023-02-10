from __future__ import print_function
import numpy as np
import os
import sys
sys.path.append('/data/lixin2021/Projects/Gr_Project')
import torch
from torch._C import device
import torch.nn as nn
from UDM2022.UDM.UDM_submit.cnn_encoder import CNN_Encoder
from UDM2022.UDM.UDM_submit.universality_extractor import U_E
from UDM2022.UDM.UDM_submit.distinction_extractor import D_E
from UDM2022.UDM.UDM_submit.loss.pin_dtw_loss import Pin_DTW


#lstm参数
hidden_size = 128 #64
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UDM_model(nn.Module):
    def __init__(self, timestep, n_inputs, hidden_size):

        super(UDM_model, self).__init__()
        
        self.dropout = 0.4
        self.hidden_size = hidden_size
        self.timestep = timestep

        # convolutional component
        self.encoder = CNN_Encoder(n_inputs, hidden_size)

        # Universality extracting
        self.U_E = U_E(self.timestep, self.dropout, self.hidden_size, device)

        # Distinction capture
        self.padding = 0 # 0 or 1
        self.D_E = D_E(c_out=self.timestep, out_len=self.timestep, d_model=hidden_size)

        self.pred_Wk = nn.ModuleList([nn.Linear(hidden_size, 1)])
        self.pin_dtw = Pin_DTW(alpha=0.5, tau=0.6, theta=0.8, gamma=1.0)
        self.loss_mse = torch.nn.MSELoss(reduction='mean')


    def forward(self, x, y, x_mark, y_mark, flag):
        actual = y
        E_ = self.encoder(x)
        O_ = self.U_E.forward(E_)

        label_len = self.timestep
        seq_y = torch.cat([E_[:, -label_len:, :], O_], 1)
        seq_y_mark = torch.cat([x_mark[:, -label_len:, :], y_mark], 1)
        if self.padding==0:
            dec_x = torch.zeros([seq_y.shape[0], self.timestep, seq_y.shape[-1]]).float().to(device)
        elif self.padding==1:
            dec_x = torch.ones([seq_y.shape[0], self.timestep, seq_y.shape[-1]]).float().to(device)
        dec_x = torch.cat([seq_y[:,:label_len,:], dec_x], dim=1).float()
        Dec_y = self.D_E(E_, x_mark, dec_x, seq_y_mark)
        Dec_y = Dec_y[:,-self.timestep:,-1].to(device)
        prediction = Dec_y.unsqueeze(2)

        mse = self.loss_mse(prediction, actual)
        loss = self.pin_dtw.forward(prediction, actual, device=device)
        if flag == True:
            return loss, mse
        return loss
        

    def predict(self, x, x_mark, y_mark):
        E_ = self.encoder(x)
        O_ = self.U_E.forward(E_)

        label_len = self.timestep
        seq_y = torch.cat([E_[:, -label_len:, :], O_], 1)
        seq_y_mark = torch.cat([x_mark[:, -label_len:, :], y_mark], 1)
        if self.padding==0:
            dec_x = torch.zeros([seq_y.shape[0], self.timestep, seq_y.shape[-1]]).float().to(device)
        elif self.padding==1:
            dec_x = torch.ones([seq_y.shape[0], self.timestep, seq_y.shape[-1]]).float().to(device)
        dec_x = torch.cat([seq_y[:,:label_len,:], dec_x], dim=1).float()
        Dec_y = self.D_E(E_, x_mark, dec_x, seq_y_mark)
        Dec_y = Dec_y[:,-self.timestep:,-1].to(device)
        prediction = Dec_y.unsqueeze(2)

        return prediction
    

    def get_pred(self, output):
        for i in range(len(self.pred_Wk)):
            temp_output = self.pred_Wk[i](output)
            output = temp_output
            
        return output



