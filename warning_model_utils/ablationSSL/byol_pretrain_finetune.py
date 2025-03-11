import argparse

import scipy
import torchvision.transforms as transforms_ecg
import ast
import copy
import h5py
import pickle

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

import torch.distributed as dist

from torch import nn
from torch.nn import functional as F

from ByolModel import CustomBYOL
from ecg_mae_pretrain_new import normalize_linear

method="byol"

class byolConfig:

    # 信号参数
    target_sample_rate = 100
    input_signal_len = 1280
    input_signal_chls = 1
    # 训练时的batch大小
    batch_size = 256
    lr = 5e-4
    min_lr = 1e-5
    max_epoch = 100

    # 微调参数
    finetune_ckp = 99
    cls_high_lr = 5e-4
    cls_low_lr = 5e-4
    cls_min_lr = 1e-5
    cls_epoch = 10
    cls_batch_size = 128

    def __init__(self, model_name="convnextv2_atto", dataSource='ptb-xl'):
        self.model_name = model_name
        self.dataSource = dataSource

        # 存储参数
        self.output_dir = "/media/lzy/Elements SE/early_warning/pretrain_result/{}/".format(method)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


# config = byolConfig()

import matplotlib.pyplot as plt
from timeseries_transformations import GaussianNoise, RandomResizedCrop, ChannelResize, Negation, \
    DynamicTimeWarp, DownSample, TimeWarp, TimeOut, ToTensor, BaselineWander, PowerlineNoise, EMNoise, BaselineShift, \
    TGaussianNoise, TRandomResizedCrop, TChannelResize, TNegation, TDynamicTimeWarp, TDownSample, TTimeOut, \
    TBaselineWander, TPowerlineNoise, TEMNoise, TBaselineShift, TGaussianBlur1d, TNormalize, Transpose


def transformations_from_strings(transformations, t_params):
    if transformations is None:
        return [ToTensor()]
    def str_to_trafo(trafo):
        if trafo == "RandomResizedCrop":
            return TRandomResizedCrop(crop_ratio_range=t_params["rr_crop_ratio_range"], output_size=t_params["output_size"])
        elif trafo == "ChannelResize":
            return TChannelResize(magnitude_range=t_params["magnitude_range"])
        elif trafo == "Negation":
            return TNegation()
        elif trafo == "DynamicTimeWarp":
            return TDynamicTimeWarp(warps=t_params["warps"], radius=t_params["radius"])
        elif trafo == "DownSample":
            return TDownSample(downsample_ratio=t_params["downsample_ratio"])
        elif trafo == "TimeWarp":
            return TimeWarp(epsilon=t_params["epsilon"])
        elif trafo == "TimeOut":
            return TTimeOut(crop_ratio_range=t_params["to_crop_ratio_range"])
        elif trafo == "GaussianNoise":
            return TGaussianNoise(scale=t_params["gaussian_scale"])
        elif trafo == "BaselineWander":
            return TBaselineWander(Cmax=t_params["bw_cmax"])
        elif trafo == "PowerlineNoise":
            return TPowerlineNoise(Cmax=t_params["pl_cmax"])
        elif trafo == "EMNoise":
            return TEMNoise(Cmax=t_params["em_cmax"])
        elif trafo == "BaselineShift":
            return TBaselineShift(Cmax=t_params["bs_cmax"])
        elif trafo == "GaussianBlur":
            return TGaussianBlur1d()
        elif trafo == "Normalize":
            return TNormalize()
        else:
            raise Exception(str(trafo) + " is not a valid transformation")

    # for numpy transformations
    # trafo_list = [str_to_trafo(trafo)
    #               for trafo in transformations] + [ToTensor()]

    # for torch transformations
    trafo_list = [ToTensor(transpose_data=False)] + [Transpose()]+ [str_to_trafo(trafo)
                                                     for trafo in transformations]+ [Transpose()]
    return trafo_list

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


class SimCLRDataTransform(object):
    def __init__(self, transform):
        if transform is None:
            self.transform = lambda x: x
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class maeDataset(torch.utils.data.Dataset):
    def __init__(self, ecgs, ):
        t_params = {"gaussian_scale": 0.005, "rr_crop_ratio_range": [0.5, 1.0],
                    "output_size": 250, "warps": 3, "radius": 10,
                    "epsilon": 10, "magnitude_range": [0.5, 2],
                    "downsample_ratio": 0.2, "to_crop_ratio_range": [0.2, 0.4],
                    "bw_cmax": 0.1, "em_cmax": 0.5, "pl_cmax": 0.2, "bs_cmax": 1}
        transformations = ["RandomResizedCrop", "TimeOut"]
        self.dataSet = torch.from_numpy(np.array(ecgs)).unsqueeze(1)
        self.n_data = len(ecgs)
        self.transformations = transformations_from_strings(
            transformations, t_params)
        self._get_simclr_pipeline_transform()
        print(self.n_data)

    def __getitem__(self, idx):

        sample = (self.dataSet[idx], 0)
        if (isinstance(self.transforms, list)):  # transforms passed as list
            for t in self.transforms:
                sample = t(sample)
        elif (self.transforms is not None):  # single transform e.g. from torchvision.transforms.Compose
            sample = self.transforms(sample)

        return sample
    def _get_simclr_pipeline_transform(self):
        data_aug = transforms_ecg.Compose(self.transformations)
        self.transforms = SimCLRDataTransform(data_aug)
    def __len__(self):
        return self.n_data


def seed_torch(seed=1029):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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


if __name__ == "__main__":

    seed_torch(111)
    parser = argparse.ArgumentParser(description="Byol for ECG")
    parser.add_argument("--flag", type=int, default=0)
    parser.add_argument('--device', default='cuda:0', help='Device', choices=['cuda:0', 'cuda:1', 'cpu'])
    parser.add_argument('--finetune_exp', type=str, default="ptb-xl",
                        help='dataset',
                        choices=["ptb-xl", "hospital", "icbeb"])
    parser.add_argument('--model_name', type=str, default="vit",
                        help='base model', choices=["vit"])
    parser.add_argument("--finetune_ckp", type=int, default=10)
    args = parser.parse_args()
    print(args)

    flag = args.flag
    model_name = args.model_name
    exp_name = args.finetune_exp

    DEVICE = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    config = byolConfig(model_name=model_name, dataSource=exp_name)
    config.finetune_ckp = args.finetune_ckp

    if flag == 0:

        DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # 载入数据集
        trainData, testData = getPretrainData()


        model = CustomBYOL(encoder=config.model_name).to(DEVICE)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('number of params: {} M'.format(n_parameters / 1e6))
        dataloader = DataLoader(maeDataset(trainData), batch_size=config.batch_size, shuffle=True)

        dataloader_test = DataLoader(maeDataset(testData), batch_size=config.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-8)
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
                byol_loss = model(data_source, DEVICE)

                loss = byol_loss

                loss.backward()
                optimizer.step()
                global_step = len_dataloader * (epoch - 1) + i + 1
                max_step = len_dataloader * config.max_epoch
                model.on_train_batch_end(global_step, max_step)
                total_loss_train += loss.item()

                loop.set_description(f'Epoch [{epoch}/{config.max_epoch}]')
                loop.set_postfix(byol_loss=byol_loss.item())
            scheduler.step()
            loss_train = total_loss_train / len_dataloader
            print('Training set:')
            print('Epoch: ' + str(epoch) + ', Loss: ' + str(loss_train))
            train_loss.append(loss_train)
            model.eval()

            pred_loss = 0
            with torch.no_grad():

                for _, ecg in enumerate(dataloader_test):

                    loss = model(ecg, DEVICE)
                    pred_loss += loss.item()
            pred_loss = pred_loss / len(dataloader_test)



            val_loss.append(pred_loss)


            if pred_loss < best_performance:
                best_performance = pred_loss
                print('best_performance: {:.4f}'.format(best_performance))
                if not os.path.exists(config.output_dir):
                    os.makedirs(config.output_dir)

                torch.save(model.state_dict(),
                           config.output_dir + f"{method}_pretrain_{epoch}.pth")


        plt.figure(2)
        plt.plot(train_loss, "r")
        plt.plot(val_loss, "b")
        plt.show()

