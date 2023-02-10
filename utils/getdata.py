import torch
import json
import os
import csv
import datetime
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset

from .Informer.timefeatures import time_features
 

# filepath


#out_len = 7
#input_len = 56



class My_dataset(Dataset):
    def __init__(self, X, y, scaler):
        super().__init__()
        self.src, self.trg = [], []
        self.src_mark, self.trg_mark = [], []
        self.freq = 'd'
        self.recscaler = scaler
        self.__zscore_norm__(X, y)
        #self.__minmax_norm__(X,y)
        #self.__nonorm__(X,y)
        
        '''
        kfold是分id存储,不用这个文件
        创建一个新函数用于返回test_dataset和train_dataset
        在新函数里按行读取json字典,如果watch0,flag为True,传入My_dataset构造函数,则只读取watch0构建test_dataset;
        否则flag为False,跳过watch0合并每一行,传入My_dataset构造函数,构建train_dataset;
        X和y分别存为一个array并进行转置,对应分别为src和trg
        归一化操作,获取src的min,max,对src和trg进行归一化
        '''

           
    def __getitem__(self, index):
        return self.src[index], self.trg[index], self.src_mark[index], self.trg_mark[index]

    def __len__(self):
        return len(self.src)
    
    def __timestamp__(self, input_end_date, seq_len, pred_len):
        Xstamp, ystamp = [], []
        for s_i in range(seq_len):
            Xstamp.append((input_end_date - datetime.timedelta(days=seq_len - 1 - s_i)).strftime('%Y-%m-%d'))
            #Xstamp.append(input_end_date - datetime.timedelta(days=seq_len - 1 - s_i))
        for p_i in range(pred_len):
            ystamp.append((input_end_date + datetime.timedelta(days=p_i + 1)).strftime('%Y-%m-%d'))
            #ystamp.append(input_end_date + datetime.timedelta(days=p_i + 1))

        Xstamp = pd.DataFrame(Xstamp, columns=['date'])
        Xstamp['date'] = pd.to_datetime(Xstamp.date)
        Xstamp = time_features(Xstamp, timeenc=0, freq=self.freq)

        ystamp = pd.DataFrame(ystamp, columns=['date'])
        ystamp['date'] = pd.to_datetime(ystamp.date)
        ystamp = time_features(ystamp, timeenc=0, freq=self.freq)
        return Xstamp, ystamp

    
    def __zscore_norm__(self, X, y):
        for i in range(len(X)):
            input_end_date = datetime.datetime.strptime(X[i][0], '%Y-%m-%d')
            Xstamp, ystamp = self.__timestamp__(input_end_date, len(X[i][1][0]), len(y[i]))
            for list in X[i][1]:
                Xlist, ylist = [], []
                for ix in list:
                    Xlist.append(float((ix-self.recscaler['mean'])/self.recscaler['std']))
                for iy in y[i]:
                    ylist.append(float((iy-self.recscaler['mean'])/self.recscaler['std']))
                self.src.append(torch.FloatTensor(Xlist))
                self.src_mark.append(Xstamp)
                self.trg.append(torch.FloatTensor(ylist))
                self.trg_mark.append(ystamp)

    def __minmax_norm__(self, X, y):
        for i in range(len(X)):
            imin, imax = min(X[i]), max(X[i])
            if imin == imax: scale = 1
            else: scale = imax-imin
            self.recscaler.append([imin, imax])

            Xlist, ylist = [], []
            for ix in X[i]:
                Xlist.append(float((ix-imin)/scale))
            for iy in y[i]:
                ylist.append(float((iy-imin)/scale))
            self.src.append(torch.FloatTensor(Xlist))
            self.trg.append(torch.FloatTensor(ylist))

    def __nonorm__(self, X, y):
        for i in range(len(X)):            
            Xlist, ylist = [], []
            for ix in X[i]:
                Xlist.append(float(ix))
            for iy in y[i]:
                ylist.append(float(iy))
            self.src.append(torch.FloatTensor(Xlist))
            self.trg.append(torch.FloatTensor(ylist))


'''def normalization(inputData, rec_scaler):
    newData = []
    for i,i_data in enumerate(inputData):
        i_min, i_max = rec_scaler[0], rec_scaler[1]
        new_data = []
        for j in range(len(i_data)):
            if i_max == i_min:
                scale = 1
            else:
                scale = i_max-i_min
            new_data.append(float((i_data[j]-i_min)/scale))
            #i_data[j] = float((i_data[j]-i_min)/scale)
    newData.append(new_data)
    return newData

def back_normalization(predData, rec_scaler):
    for i,i_data in enumerate(predData):
        i_min, i_max = rec_scaler[i][0], rec_scaler[i][1]
        for j in range(len(i_data)):
            i_data[j] = torch.FloatTensor(float(i_data[j]*(i_max-i_min) + i_min))
    return predData'''

