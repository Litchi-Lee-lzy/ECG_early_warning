import ast
import copy
import h5py
import pickle

import scipy
import wfdb
from scipy.signal import resample
import torch
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

from cpcModel import cpcPretrainModel
from ecg_mae_pretrain_new import normalize_linear


class cpcConfig:
    # 编码器：
    model_name = "vit"

    # 信号参数
    target_sample_rate = 100
    input_signal_len = 1280
    input_signal_chls = 1
    # 训练时的batch大小
    batch_size = 128
    lr = 1e-3
    min_lr = 1e-5
    max_epoch = 100

    # 微调参数
    cls_high_lr = 1e-3
    cls_low_lr = 1e-3
    cls_min_lr = 1e-4
    cls_epoch = 10
    cls_batch_size = 128
    dataSource = 'ptb-xl'
    num_classes = 3


    # 存储参数
    output_dir = "/media/lzy/Elements SE/early_warning/pretrain_result/cpc/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



config = cpcConfig()

import matplotlib.pyplot as plt
def plot_maskECG(signal, patches,pred_pixel_values ,masked_indices,epoch = None):

    signal = signal.detach().cpu().numpy()
    masked_indices = masked_indices.detach().cpu().numpy().reshape(-1)
    print(masked_indices)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    ax.set_xticks(np.arange(0, 10, 0.5))
    ax.set_yticks(np.arange(-22.5, +1.0, 0.5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.minorticks_on()
    x_major_locator = MultipleLocator(1)  # 设置 x 轴主要刻度线每隔 1 个单位显示一个
    ax.xaxis.set_minor_locator(x_major_locator)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.grid(axis='x', which='major', linestyle='--', linewidth='0.2', color='red')
    #     ax.grid(which='minor', linestyle='-', linewidth='0.1', color=(1, 0.7, 0.7))
    # 循环绘制每个通道的曲线
    t = np.arange(0, 1000 * 1 / 100, 1 / 100)  # 时间点
    for i in range(12):
        # 获取当前通道的数据和掩码
        channel_data = signal[i]
        print(len(channel_data))
        channel_mask = masked_indices[i]

        # 绘制当前通道的曲线
        ax.plot(t, channel_data - i * 2, c='k', lw=0.5)
        for ind_mask in masked_indices:
            start_ind = ind_mask * config.vit_patch_length
            end_ind = (ind_mask + 1) * config.vit_patch_length
            ax.plot(t[start_ind : end_ind], channel_data[start_ind : end_ind] - i * 2, c='r', lw=0.8)

    ax.set_xlim(left=0, right=10)
    ax.set_ylim(top=1, bottom=-22.5)

    plt.show()
    pred_pixel_values = pred_pixel_values.detach().squeeze().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    ax.set_xticks(np.arange(0, 10, 0.5))
    ax.set_yticks(np.arange(-22.5, +1.0, 0.5))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.minorticks_on()
    x_major_locator = MultipleLocator(1)  # 设置 x 轴主要刻度线每隔 1 个单位显示一个
    ax.xaxis.set_minor_locator(x_major_locator)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    ax.grid(axis='x', which='major', linestyle='--', linewidth='0.2', color='red')
    #     ax.grid(which='minor', linestyle='-', linewidth='0.1', color=(1, 0.7, 0.7))
    # 循环绘制每个通道的曲线
    t = np.arange(0, 1000 * 1 / 100, 1 / 100)  # 时间点
    for j in range(100):
        start_ind = j * config.vit_patch_length
        end_ind = (j + 1) * config.vit_patch_length
        for i in range(12):
            # 获取当前通道的数据和掩码

            channel_data = pred_pixel_values[j, i, :]
            # 绘制当前通道的曲线
            ax.plot(t[start_ind:end_ind], channel_data - i * 2, c='k', lw=0.5)
    ax.set_xlim(left=0, right=10)
    ax.set_ylim(top=1, bottom=-22.5)

    plt.show()



def ECGplot_multiLeads(data, fig_name=None):
    '''绘制多导联'''
    # print(data)
    # data = np.random.rand(12, 5000)
    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    ax.set_xticks(np.arange(0, 10.5, 0.2))
    ax.set_yticks(np.arange(-23, +1.0, 0.5))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
    # 隐藏 x 和 y 轴刻度标签的数字
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #   # 设置 x 轴次要刻度线
    ax.minorticks_on()
    x_major_locator = MultipleLocator(0.04)  # 设置 x 轴主要刻度线每隔 1 个单位显示一个
    ax.xaxis.set_minor_locator(x_major_locator)
    ax.grid(which='major', linestyle='-', linewidth='0.3', color='gray')
    ax.grid(which='minor', linestyle='-', linewidth='0.1', color=(1, 0.7, 0.7))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(2))

    t = np.arange(0, len(data[0]) * 1 / 100, 1 / 100)
    lead = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    # lead = ["I","II","III","aVR","aVL","aVF","V1","V2"]
    for i, l in enumerate(lead):
        ax.plot(t, np.array(data[i]) - 2 * i, label=l, linewidth=0.8, color='black')

    # ymin = -1.5
    # ymax = 1.5
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.set(xlabel='time (s)')
    #     ax.set(ylabel='Voltage (mV)')
    #     ax.autoscale(tight=True)
    ax.set_xlim(left=0, right=10.5)
    ax.set_ylim(top=1, bottom=-23)
    plt.show()




def getPretrainData():
    path = '/home/lzy/workspace/dataSet/physionet.org/files/ptb-xl/1.0.2/'
    Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    X = []
    for ind, row in Y.iterrows():
        label = dict(row.scp_codes)
        # if label.keys() == {"NORM", "SR"}:
        X.append(path + row.filename_hr)
    ecgs = []
    lead_list = [0]
    for lead in lead_list:
        for i in X:
            ecgData = wfdb.rdsamp(i)[0]
            # for lead in [1]:
            ecgSig = ecgData[:, lead]
            # ecgSig = IIRRemoveBL(ecgSig, 500, Fc=0.67)
            ecgSig = scipy.signal.resample(np.array(ecgSig).reshape(5000), 1280)
            # ecgSig = median_filter(ecgSig)
            ecgSig = normalize_linear(ecgSig.reshape(1, 1280)).reshape(1280)
            ecgs.append(ecgSig)
        print("PTB-XL: ", end="  ")
        print(len(X))
        # chapman G12EC
        with h5py.File("/media/lzy/Elements SE/ECG_self_supervised/NINGBO.hdf5", 'r') as f:
            ecgSig = np.array(f['data'])
            print("chapman: ", end="  ")
            print(len(ecgSig))
            for ecgData in ecgSig:
                ecgSig = ecgData[lead, :]
                # ecgSig = IIRRemoveBL(ecgSig, 100, Fc=0.67)
                ecgSig = scipy.signal.resample(np.array(ecgSig).reshape(1000), 1280)
                # ecgSig = median_filter(ecgSig)
                ecgSig = normalize_linear(ecgSig.reshape(1, 1280)).reshape(1280)
                ecgs.append(ecgSig)
        with h5py.File("/media/lzy/Elements SE/ECG_self_supervised/G12EC.hdf5", 'r') as f:
            ecgSig = np.array(f['data'])
            print("Georgia: ", end="  ")
            print(len(ecgSig))
            for ecgData in ecgSig:
                ecgSig = ecgData[lead, :]
                # ecgSig = IIRRemoveBL(ecgSig, 100, Fc=0.67)
                ecgSig = scipy.signal.resample(np.array(ecgSig).reshape(1000), 1280)
                # ecgSig = median_filter(ecgSig)
                ecgSig = normalize_linear(ecgSig.reshape(1, 1280)).reshape(1280)
                ecgs.append(ecgSig)

    trainData, testData  = train_test_split(ecgs, test_size=0.1, random_state=111)
    # valData, testData = train_test_split(testData, test_size=0.5)
    print(len(X))
    return trainData, testData,


class maeDataset(torch.utils.data.Dataset):
    def __init__(self, ecgs):

        self.dataSet = torch.from_numpy(np.array(ecgs))
        self.n_data= len(ecgs)
        print(self.n_data)
    def __getitem__(self, item):

        return self.dataSet[item]

    def __len__(self):
        return self.n_data


def seed_torch(seed=1029):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True




if __name__ == "__main__":
    # CombiningDatasetAndVisualization()
    seed_torch(111)
    flag = 0

    if flag == 0:

        DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        # 载入数据集
        trainData, testData = getPretrainData()


        model = cpcPretrainModel(encoder=config.model_name).to(DEVICE)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('number of params: {} M'.format(n_parameters / 1e6))
        dataloader = DataLoader(maeDataset(trainData), batch_size=256, shuffle=True)

        dataloader_test = DataLoader(maeDataset(testData), batch_size=256, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch, eta_min=config.min_lr)


        val_loss = []
        train_loss = []
        best_performance = 100000
        for epoch in range(1, config.max_epoch + 1):
            model.train()
            len_dataloader = len(dataloader)
            loop = tqdm(enumerate(dataloader), desc="Training", position=0)
            total_loss_train = 0
            for i, data_source in loop:
                optimizer.zero_grad()
                ecg = data_source.type(torch.FloatTensor)
                ecg = ecg.to(DEVICE)

                cpc_loss = model(ecg)

                loss = cpc_loss

                loss.backward()
                optimizer.step()
                total_loss_train += loss.item()

                loop.set_description(f'Epoch [{epoch}/{config.max_epoch}]')
                loop.set_postfix(cpc_loss=cpc_loss.item())
            scheduler.step()
            loss_train = total_loss_train / len_dataloader
            print('Training set:')
            print('Epoch: ' + str(epoch) + ', Loss: ' + str(loss_train))
            train_loss.append(loss_train)
            model.eval()

            pred_loss = 0
            with torch.no_grad():

                for _, ecg in enumerate(dataloader_test):
                    ecg = ecg.type(torch.FloatTensor).to(DEVICE)
                    loss = model(ecg)
                    pred_loss += loss.item()
            pred_loss = pred_loss / len(dataloader_test)



            val_loss.append(pred_loss)


            if pred_loss < best_performance:
                best_performance = pred_loss
                print('best_performance: {:.4f}'.format(best_performance))
                if not os.path.exists(config.output_dir):
                    os.makedirs(config.output_dir)

                torch.save(model.state_dict(),
                           config.output_dir + f"cpc_pretrain_{epoch}.pth")


        plt.figure(2)
        plt.plot(train_loss, "r")
        plt.plot(val_loss, "b")
        plt.show()

