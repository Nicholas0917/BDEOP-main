import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy import linalg as la
from scipy import special as ss
from utils.op import transition

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HiPPO_LegT(nn.Module):
    def __init__(self, N, dt=1.0, discretization='bilinear'):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super(HiPPO_LegT,self).__init__()
        self.N = N
        A, B = transition('lmu', N)
        C = np.ones((1, N))
        D = np.zeros((1,))
        # dt, discretization options
        A, B, _, _, _ = signal.cont2discrete((A, B, C, D), dt=dt, method=discretization)

        B = B.squeeze(-1)

        self.register_buffer('A', torch.Tensor(A).to(device)) 
        self.register_buffer('B', torch.Tensor(B).to(device)) 
        vals = np.arange(0.0, 1.0, dt)
        self.register_buffer('eval_matrix',  torch.Tensor(
            ss.eval_legendre(np.arange(N)[:, None], 1 - 2 * vals).T).to(device))
        # print("eval_matrix shape:", self.eval_matrix.shape)# [length, N]
    
    def forward(self, inputs):  # torch.Size([128, 1, 1]) -
        """
        inputs : (length, ...)
        output : (length, ..., N) where N is the order of the HiPPO projection
        """

        c = torch.zeros(inputs.shape[:-1] + tuple([self.N])).to(device)  # torch.Size([1, 256])
        cs = []

        for f in inputs.permute([-1, 0, 1]):
            f = f.unsqueeze(-1)
            # f: [1,1]
            new = f @ self.B.unsqueeze(0) # [B, D, N, 256]
            c = F.linear(c, self.A) + new
            # c = [1,256] * [256,256] + [1, 256]
            cs.append(c)
        return torch.stack(cs, dim=0)

    def reconstruct(self, c):
        a = (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)
        return (self.eval_matrix @ c.unsqueeze(-1)).squeeze(-1)

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

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len,  modes1, compression=0, ratio=0.5, mode_type=0, use_amp=True):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.compression = compression
        self.mode_type = mode_type
        self.use_amp = use_amp

        modes2 = modes1
        self.modes2 = min(modes2, seq_len//2)
        self.index0 = list(range(0, int(ratio*min(seq_len//2, modes2))))
        self.index1 = list(range(len(self.index0), self.modes2))
        np.random.shuffle(self.index1)
        self.index1 = self.index1[:min(seq_len//2, self.modes2)-int(ratio * min(seq_len//2, modes2))]
        self.index = self.index0 + self.index1
        self.index.sort()

        self.scale = (1 / (in_channels * out_channels))
        print('mode_Num:', self.modes2)
        
        if self.use_amp:
            self.weights1_real = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
            self.weights1_imag = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.float))
        else:
            self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, len(self.index), dtype=torch.cfloat))

    def forward(self, x):
        B, D, L = x.shape
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        # print("before FFT:", x.shape)
        x_ft = torch.fft.rfft(x)
        if self.use_amp:
            x_ft_real = x_ft.real
            x_ft_imag = x_ft.imag
        # print("before FFT:", x_ft.shape)
        # Multiply relevant Fourier modes
        if self.use_amp:
            out_ft_real = torch.zeros(B, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.float)
            out_ft_imag = torch.zeros(B, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.float)
        else:
            out_ft = torch.zeros(B, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        # a = x_ft[:, :self.modes1]
        # out_ft[:, :self.modes1] = torch.einsum("bix,iox->box", a, self.weights1)
        if self.modes1 > 1000:
            for wi, i in enumerate(self.index):
                if self.use_amp:
                    out_ft_real[:, :, i] = torch.einsum('bi,io->bo',(x_ft_real[:, :, i], self.weights1_real[:, wi]))
                    out_ft_imag[:, :, i] = torch.einsum('bi,io->bo',(x_ft_imag[:, :, i], self.weights1_imag[:, wi]))
                else:
                    out_ft[:, :, i] = torch.einsum('bi,io->bo',(x_ft[:, :, i], self.weights1[:, wi]))
        else:
            if self.use_amp:
                a_real = x_ft_real[:, :, :self.modes2]
                a_imag = x_ft_imag[:, :, :self.modes2]
                out_ft_real[:, :, :self.modes2] = torch.einsum("bix,iox->box",(a_real, self.weights1_real))
                out_ft_imag[:, :, :self.modes2] = torch.einsum("bix,iox->box",(a_imag, self.weights1_imag))
            else:
                a = x_ft[:, :,:, :self.modes2]
                out_ft[:, :, :self.modes2] = torch.einsum("bix,iox->box", a, self.weights1)
        
        # Return to physical space
        if self.use_amp:
            out_ft = torch.complex(out_ft_real, out_ft_imag)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs, N=512):
        super(Model, self).__init__()
        # basic configuration
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.output_attention = configs.output_attention
        
        # Decomp part
        if self.configs.Decomp == 1:
            print("Using decompression!")
            kernel_size = 25
            self.decompsition = series_decomp(kernel_size)
            if self.configs.individual:
                self.Linear_Trend = nn.ModuleList()
                for i in range(self.channels):
                    self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))
                    self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            else:
                self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
                self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        elif self.configs.Decomp  == 2:
            print("Using decompression * 2!")
            self.decompsition_1 = series_decomp(13)
            self.decompsition_2 = series_decomp(25)
            if self.configs.individual:
                self.Linear_Trend_1 = nn.ModuleList()
                self.Linear_Trend_2 = nn.ModuleList()
                for i in range(self.channels):
                    self.Linear_Trend_1.append(nn.Linear(self.seq_len,self.pred_len))
                    self.Linear_Trend_1[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                    self.Linear_Trend_2.append(nn.Linear(self.seq_len,self.pred_len))
                    self.Linear_Trend_2[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            else:
                self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
                self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        elif self.configs.Decomp  == 3:
            print("Using decompression * 3!")
            self.decompsition_1 = series_decomp(13)
            self.decompsition_2 = series_decomp(25)
            self.decompsition_3 = series_decomp(169)
            if self.configs.individual:
                self.Linear_Trend_1 = nn.ModuleList()
                self.Linear_Trend_2 = nn.ModuleList()
                self.Linear_Trend_3 = nn.ModuleList()
                for i in range(self.channels):
                    self.Linear_Trend_1.append(nn.Linear(self.seq_len,self.pred_len))
                    self.Linear_Trend_1[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                    self.Linear_Trend_2.append(nn.Linear(self.seq_len,self.pred_len))
                    self.Linear_Trend_2[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                    self.Linear_Trend_3.append(nn.Linear(self.seq_len,self.pred_len))
                    self.Linear_Trend_3[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            else:
                self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
                self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        
        # FDL part
        self.mode_Num = min(configs.mode_Num,self.pred_len//2)
        self.mode_type=configs.mode_type
        if self.configs.FDL == 1:
            print("Using FDL in the original space!")
            self.spec_conv_1 = SpectralConv1d(in_channels=self.channels, out_channels=self.channels, seq_len=self.seq_len, modes1=self.mode_Num, compression=0, ratio=configs.ratio, mode_type=configs.mode_type)
        
        self.multiscale = [1]
        # LegT part
        self.LegT_Order = configs.LegT_Order
        self.legts = nn.ModuleList([HiPPO_LegT(N=n, dt=1./self.pred_len) for n in self.LegT_Order])
        self.mlp = nn.Linear(len(self.LegT_Order), 1)
        self.linear = nn.ModuleList([nn.Linear(n, n) for n in self.LegT_Order])

        # RevIN module
        if self.configs.RevIN:
            print("Using RevIN!")
            self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))


    def forward(self, x_enc, x_mark_enc, x_dec_true, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc: [Batch, Input_L, Channel]
        
        return_data = []

        if self.configs.RevIN:
            means = x_enc.mean(1, keepdim=True).detach()
            #mean
            x_enc = x_enc - means
            #var
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
            x_enc /= stdev
            # affine
            x_enc = x_enc * self.affine_weight + self.affine_bias
        
        # return_data.append(x_enc)

        if self.configs.Decomp == 1:
            seasonal_init, trend_init = self.decompsition(x_enc)
            # trend calculation
            trend_init = trend_init.permute(0,2,1)
            if self.configs.individual:
                trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(trend_init.device)
                for i in range(self.channels):
                    trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
            else:
                trend_output = self.Linear_Trend(trend_init)
            trend_output = trend_output.permute(0,2,1)

        elif self.configs.Decomp == 2:
            seasonal_init, trend_init_1 = self.decompsition_1(x_enc)
            seasonal_init, trend_init_2 = self.decompsition_2(seasonal_init)
            # trend calculation
            trend_init_1 = trend_init_1.permute(0,2,1)
            trend_init_2 = trend_init_2.permute(0,2,1)
            if self.configs.individual:
                trend_output_1 = torch.zeros([trend_init_1.size(0), trend_init_1.size(1), self.pred_len], dtype=trend_init_1.dtype).to(trend_init_1.device)
                trend_output_2 = torch.zeros([trend_init_2.size(0), trend_init_2.size(1), self.pred_len], dtype=trend_init_2.dtype).to(trend_init_2.device)
                for i in range(self.channels):
                    trend_output_1[:,i,:] = self.Linear_Trend_1[i](trend_init_1[:,i,:])
                    trend_output_2[:,i,:] = self.Linear_Trend_2[i](trend_init_2[:,i,:])
            else:
                trend_output = self.Linear_Trend(trend_init)
            trend_output_1 = trend_output_1.permute(0,2,1)
            trend_output_2 = trend_output_2.permute(0,2,1)

        elif self.configs.Decomp == 3:
            seasonal_init, trend_init_1 = self.decompsition_1(x_enc)
            seasonal_init, trend_init_2 = self.decompsition_2(seasonal_init)
            seasonal_init, trend_init_3 = self.decompsition_3(seasonal_init)
            # trend calculation
            trend_init_1 = trend_init_1.permute(0,2,1)
            trend_init_2 = trend_init_2.permute(0,2,1)
            trend_init_3 = trend_init_3.permute(0,2,1)
            if self.configs.individual:
                trend_output_1 = torch.zeros([trend_init_1.size(0), trend_init_1.size(1), self.pred_len], dtype=trend_init_1.dtype).to(trend_init_1.device)
                trend_output_2 = torch.zeros([trend_init_2.size(0), trend_init_2.size(1), self.pred_len], dtype=trend_init_2.dtype).to(trend_init_2.device)
                trend_output_3 = torch.zeros([trend_init_3.size(0), trend_init_3.size(1), self.pred_len], dtype=trend_init_3.dtype).to(trend_init_3.device)
                for i in range(self.channels):
                    trend_output_1[:,i,:] = self.Linear_Trend_1[i](trend_init_1[:,i,:])
                    trend_output_2[:,i,:] = self.Linear_Trend_2[i](trend_init_2[:,i,:])
                    trend_output_3[:,i,:] = self.Linear_Trend_3[i](trend_init_3[:,i,:])
            else:
                trend_output = self.Linear_Trend(trend_init)
            trend_output_1 = trend_output_1.permute(0,2,1)
            trend_output_2 = trend_output_2.permute(0,2,1)
            trend_output_3 = trend_output_3.permute(0,2,1)
        
        else:
            seasonal_init = x_enc
        
        # sensonal calculation
        x_decs = []
        
        for i in range(0, len(self.LegT_Order)):
            # print('input to FDL:\n', seasonal_init.permute([0, 2, 1]).shape)# B, D, L
            if self.configs.FDL == 1:
                x_in = self.spec_conv_1(seasonal_init.permute([0, 2, 1]))
            # print('input after FDL:\n', x_in.shape)# B, D, L
            
            # print('input to legt:\n', x_in.shape)# B, D, L
            legt = self.legts[i]
            x_in_c = legt(x_in).permute([1, 2, 3, 0])
            # print('input after legt:\n', x_in_c.shape)# B, D, N, L
            
            x_in_c = x_in_c.permute([0, 1, 3, 2])
            out1 = self.linear[i](x_in_c).permute([0, 1, 3, 2])

            if self.seq_len >= self.pred_len:
                x_dec_c = out1.transpose(2, 3)[:,:, self.pred_len-1, :]
            else:
                x_dec_c = out1.transpose(2, 3)[:,:, -1, :]
            
            # print('output before legt_R:\n', x_dec_c.shape)
            x_dec = x_dec_c @ (legt.eval_matrix[-self.pred_len:,:].T)
            # print('output:\n', x_dec.shape)

            x_decs += [x_dec]
        # print('output:\n', x_decs[0].shape)
        seasonal_output = self.mlp(torch.stack(x_decs, dim=-1)).squeeze(-1).permute(0,2,1)
        # print('output shape:\n', seasonal_output.shape)
        
        if self.configs.Decomp == 1:
            x = seasonal_output + trend_output
            return_data.append(trend_output)
            return_data.append(seasonal_output)
        
        elif self.configs.Decomp == 2:
            x = seasonal_output + trend_output_1 + trend_output_2
            return_data.append(trend_output_1)
            return_data.append(trend_output_2)
            return_data.append(seasonal_output)
        
        elif self.configs.Decomp == 3:
            x = seasonal_output + trend_output_1 + trend_output_2 + trend_output_3
            return_data.append(trend_output_1)
            return_data.append(trend_output_2)
            return_data.append(trend_output_3)
            return_data.append(seasonal_output)
        
        else:
            x = seasonal_output
            return_data.append(seasonal_output)
        
        return_data = torch.stack(return_data, 0)

        if self.configs.RevIN:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-10)
            x = x * stdev
            x = x + means
            if self.output_attention:
                return_data[0] = return_data[0] - self.affine_bias
                return_data = return_data / (self.affine_weight + 1e-10)
                return_data = return_data * stdev
                return_data[0] = return_data[0] + means

        if self.output_attention:
            return x, return_data
        else:
            return x # [B, L, D]