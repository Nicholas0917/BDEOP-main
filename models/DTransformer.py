from re import X
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import Transformer

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    DLinear-LSTM
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        self.Linear_Seasonal = Transformer.Model(configs).float()
        self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

    def forward(self, x, x_mark_enc, x_dec_true, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        dec_inp = torch.zeros_like(x_dec_true[:, -self.pred_len:, :]).float()
        
        seasonal_dec_inp = torch.cat([seasonal_init[:, -self.label_len:, :], dec_inp], dim=1).float()
        seasonal_output = self.Linear_Seasonal(seasonal_init, x_mark_enc, seasonal_dec_inp, x_mark_dec)

        trend_init = trend_init.permute(0,2,1)
        trend_output = self.Linear_Trend(trend_init).permute(0,2,1)

        x = seasonal_output + trend_output
        return x # to [Batch, Output length, Channel]
