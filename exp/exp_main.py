from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, FEDformer, DLinear, FiLM, DAutoformer, DTransformer, DFiLM, DFEDformer, BDEOP
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import random
warnings.filterwarnings('ignore')

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

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'FEDformer': FEDformer,
            'FiLM': FiLM,
            'DLinear': DLinear,
            'DAutoformer': DAutoformer,
            'DTransformer': DTransformer,
            'DFiLM': DFiLM,
            'DFEDformer': DFEDformer,
            'BDEOP': BDEOP,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, relate) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if self.args.add_noise_train:
                    batch_x = batch_x + 0.3*torch.from_numpy(np.random.normal(0, 1, size=batch_x.shape)).float().to(self.device)
                if self.args.add_noise_vali:
                    noise = np.zeros((batch_x.shape[0], batch_x.shape[1]))
                    for j in range(batch_x.shape[0]):
                        for i in range(batch_x.shape[1] - 1):
                            noise[j, i + 1] = noise[j, i] + random.gauss(0, 0.01)
                    batch_x[:, :, -1] += torch.from_numpy(noise).float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                relate = relate.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'DLinear':
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'DLinear':
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, relate) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                
                if self.args.add_noise_train:
                    batch_x = batch_x + 0.3*torch.from_numpy(np.random.normal(0, 1, size=batch_x.float().shape)).float().to(self.device)
                if self.args.add_noise_vali:
                    noise = np.zeros((batch_x.shape[0], batch_x.shape[1]))
                    for j in range(batch_x.shape[0]):
                        for i in range(batch_x.shape[1] - 1):
                            noise[j, i + 1] = noise[j, i] + random.gauss(0, 0.01)
                    batch_x[:, :, -1] += torch.from_numpy(noise).float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                relate = relate.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'DLinear':
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.model == 'DLinear':
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if self.args.output_attention:
            criterion = self._select_criterion()
            total_loss_Seasonal = []
            total_loss_Trend1 = []
            total_loss_Trend2 = []
            total_loss_Trend3 = []
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        return_datas = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        if self.args.Decomp == 1:
            self.decompsition = series_decomp(25)
        elif self.args.Decomp  == 2:
            self.decompsition_1 = series_decomp(13)
            self.decompsition_2 = series_decomp(25)
        elif self.args.Decomp  == 3:
            self.decompsition_1 = series_decomp(13)
            self.decompsition_2 = series_decomp(25)
            self.decompsition_3 = series_decomp(169)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, relate) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                if self.args.add_noise_train:
                    batch_x = batch_x + 0.3*torch.from_numpy(np.random.normal(0, 1, size=batch_x.float().shape)).float().to(self.device)
                if self.args.add_noise_vali:
                    noise = np.zeros((batch_x.shape[0], batch_x.shape[1]))
                    for j in range(batch_x.shape[0]):
                        for i in range(batch_x.shape[1] - 1):
                            noise[j, i + 1] = noise[j, i] + random.gauss(0, 0.01)
                    batch_x[:, :, -1] += torch.from_numpy(noise).float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                relate = relate.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'DLinear':
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                                return_data = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[1]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'DLinear':
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            return_data = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[1]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if self.args.output_attention:
                    return_data = return_data.detach().cpu()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if self.args.output_attention:
                    return_datas.append(return_data.numpy())

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                    if self.args.output_attention:
                        if self.args.Decomp == 1:
                            True_Seasonal, True_Trend1 = self.decompsition(torch.from_numpy(batch_y))
                            Pred_Trend1 = return_data[0]
                            Pred_Seasonal = return_data[1]
                            visual(True_Seasonal[0, :, -1], Pred_Seasonal[0, :, -1], os.path.join(folder_path, str(i) + '_seasonal.pdf'))
                            visual(True_Trend1[0, :, -1], Pred_Trend1[0, :, -1], os.path.join(folder_path, str(i) + '_trend1.pdf'))
                            loss_Seasonal = criterion(Pred_Seasonal, True_Seasonal)
                            loss_Trend1 = criterion(Pred_Trend1, True_Trend1)
                            total_loss_Seasonal.append(loss_Seasonal)
                            total_loss_Trend1.append(loss_Trend1)
                        elif self.args.Decomp == 2:
                            True_Seasonal, True_Trend1 = self.decompsition_1(torch.from_numpy(batch_y))
                            True_Seasonal, True_Trend2 = self.decompsition_2(True_Seasonal)
                            Pred_Trend1 = return_data[0]
                            Pred_Trend2 = return_data[1]
                            Pred_Seasonal = return_data[2]
                            visual(True_Seasonal[0, :, -1], Pred_Seasonal[0, :, -1], os.path.join(folder_path, str(i) + '_seasonal.pdf'))
                            visual(True_Trend1[0, :, -1], Pred_Trend1[0, :, -1], os.path.join(folder_path, str(i) + '_trend1.pdf'))
                            visual(True_Trend2[0, :, -1], Pred_Trend2[0, :, -1], os.path.join(folder_path, str(i) + '_trend2.pdf'))
                            loss_Seasonal = criterion(Pred_Seasonal, True_Seasonal)
                            loss_Trend1 = criterion(Pred_Trend1, True_Trend1)
                            loss_Trend2 = criterion(Pred_Trend2, True_Trend2)
                            total_loss_Seasonal.append(loss_Seasonal)
                            total_loss_Trend1.append(loss_Trend1)
                            total_loss_Trend2.append(loss_Trend2)
                        elif self.args.Decomp == 3:
                            True_Seasonal, True_Trend1 = self.decompsition_1(torch.from_numpy(batch_y))
                            True_Seasonal, True_Trend2 = self.decompsition_2(True_Seasonal)
                            True_Seasonal, True_Trend3 = self.decompsition_3(True_Seasonal)
                            Pred_Trend1 = return_data[0]
                            Pred_Trend2 = return_data[1]
                            Pred_Trend3 = return_data[2]
                            Pred_Seasonal = return_data[3]
                            visual(True_Seasonal[0, :, -1], Pred_Seasonal[0, :, -1], os.path.join(folder_path, str(i) + '_seasonal.pdf'))
                            visual(True_Trend1[0, :, -1], Pred_Trend1[0, :, -1], os.path.join(folder_path, str(i) + '_trend1.pdf'))
                            visual(True_Trend2[0, :, -1], Pred_Trend2[0, :, -1], os.path.join(folder_path, str(i) + '_trend2.pdf'))
                            visual(True_Trend3[0, :, -1], Pred_Trend3[0, :, -1], os.path.join(folder_path, str(i) + '_trend3.pdf'))
                            loss_Seasonal = criterion(Pred_Seasonal, True_Seasonal)
                            loss_Trend1 = criterion(Pred_Trend1, True_Trend1)
                            loss_Trend2 = criterion(Pred_Trend2, True_Trend2)
                            loss_Trend3 = criterion(Pred_Trend3, True_Trend3)
                            total_loss_Seasonal.append(loss_Seasonal)
                            total_loss_Trend1.append(loss_Trend1)
                            total_loss_Trend2.append(loss_Trend2)
                            total_loss_Trend3.append(loss_Trend3)
        
        if self.args.output_attention:
            total_loss_Seasonal = np.average(total_loss_Seasonal)
            total_loss_Trend1 = np.average(total_loss_Trend1)
            total_loss_Trend2 = np.average(total_loss_Trend2)
            total_loss_Trend3 = np.average(total_loss_Trend3)
            print("Seasonal MSE:", total_loss_Seasonal)
            print("Trend1 MSE:", total_loss_Trend1)
            print("Trend2 MSE:", total_loss_Trend2)
            print("Trend3 MSE:", total_loss_Trend3)


        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)
        if self.args.output_attention:
            return_datas = np.array(return_datas)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, _ = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'x.npy', inputx)
        if self.args.output_attention:
            np.save(folder_path + 'intermediates.npy', return_datas)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, relate) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                relate = relate.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'DLinear':
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.model == 'DLinear':
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
