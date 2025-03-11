import ast

import h5py
import scipy
import torch
import wfdb
from sklearn.model_selection import train_test_split
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.utils.data import DataLoader
import tqdm

import numpy as np
import os
import pandas as pd

from utils.dfilters import IIRRemoveBL
from warning_model_utils.pretrain_model import Config, MAE_linearmask

config = Config()



import matplotlib.pyplot as plt
def plot_maskECG(signal, patches,pred_pixel_values ,masked_indices,epoch = None):

    signal = signal.detach().cpu().numpy().reshape(-1)
    patches = patches.detach().cpu().numpy().reshape(config.input_signal_len // config.vit_patch_length, config.vit_patch_length)

    pred_pixel_values = pred_pixel_values.detach().cpu().numpy().reshape(-1, config.vit_patch_length)
    masked_indices = masked_indices.detach().cpu().numpy().reshape(-1)
    print(masked_indices)
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    axs[0].plot(signal)
    axs[0].set_title('ecg')
    ecg2 = patches.reshape(-1)
    axs[1].plot(ecg2)
    for i in masked_indices:
        axs[1].plot(range(i * config.vit_patch_length, (i+1) * config.vit_patch_length), patches[i], "r")

    axs[1].set_title('ecg with mask')
    # for i,mask in enumerate(pred_pixel_values):
    #     patches[masked_indices[i]] = mask
    ecg3 = pred_pixel_values.reshape(-1)
    axs[2].plot(ecg3)
    axs[2].set_title('new ecg')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图形

    # plt.savefig(f"./maeResult/ModelPretrain/fig/fig_{epoch}.png")
    plt.show()
    # plt.close()
    # calculate reconstruction loss

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
    lead_list = [0, 1]
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



def normalize_linear(arr):
    # 计算数组每一行的最小值和最大值
    row_min = np.min(arr, axis=1, keepdims=True)
    row_max = np.max(arr, axis=1, keepdims=True)
    # 将数组每一行进行线性归一化
    normalize_flag = (row_max == row_min)
    row_max[normalize_flag] = 1
    row_min[normalize_flag] = 0
    arr_normalized = (arr - row_min) / (row_max - row_min)
    # 将最大最小值相等的行还原为原始值
    arr_normalized[normalize_flag.squeeze()] = arr[normalize_flag.squeeze()]

    return arr_normalized
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
    seed_torch(111)
    flag = 1

    if flag == 0:

        DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        # 载入数据集
        trainData, testData = getPretrainData()


        model = MAE_linearmask(pre_train="train").to(DEVICE)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('number of params: {} M'.format(n_parameters / 1e6))
        dataloader = DataLoader(maeDataset(trainData), batch_size=128, shuffle=True)

        dataloader_test = DataLoader(maeDataset(testData), batch_size=32, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch, eta_min=config.min_lr)


        val_loss = []
        train_loss = []
        best_performance = 100000
        for epoch in range(1, config.max_epoch + 1):
            model.train()
            len_dataloader = len(dataloader)
            loop = tqdm.tqdm(enumerate(dataloader), desc="Training")
            total_loss_train = 0
            for i, data_source in loop:
                optimizer.zero_grad()
                ecg = data_source.type(torch.FloatTensor)
                ecg = ecg.to(DEVICE)

                masked_loss, unmasked_loss = model(ecg)

                loss = 0.7 * masked_loss["loss"] + 0.3 * unmasked_loss["loss"]

                loss.backward()
                optimizer.step()
                total_loss_train += loss.item()

                loop.set_description(f'Epoch [{epoch}/{config.max_epoch}]')
                loop.set_postfix(masked_loss=masked_loss["loss"].item(), unmasked_loss=unmasked_loss["loss"].item(),
                                 masked_rec_loss=masked_loss["Reconstruction_Loss"].item())
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
                    loss, _ = model(ecg)
                    pred_loss += loss["loss"].item()
            pred_loss = pred_loss / len(dataloader_test)



            val_loss.append(pred_loss)


            if pred_loss < best_performance:
                best_performance = pred_loss
                print('best_performance: {:.4f}'.format(best_performance))
                if not os.path.exists(config.output_dir):
                    os.makedirs(config.output_dir)

                torch.save(model.state_dict(),
                           config.output_dir + f"mask_unmask_model_{epoch}.pth")
                # with torch.no_grad():
                #     model.plot = True
                #     for _, ecg in enumerate(dataloader_test):
                #         for i in range(1):
                #             sig = ecg[i].reshape(1, 1280).type(torch.FloatTensor).to(DEVICE)
                #             patches, pred_pixel_values, masked_indices = model(sig)
                #             plot_maskECG(ecg[i], patches, pred_pixel_values, masked_indices, epoch+1)
                #         break
                #     model.plot = False

        plt.figure(2)
        plt.plot(train_loss, "r")
        plt.plot(val_loss, "b")
        plt.show()

    elif flag == 1:
        DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        # 载入数据集
        # trainData, testData = getPretrainData()
        #
        # dataloader_test = DataLoader(maeDataset(testData), batch_size=32, shuffle=False)

        model = MAE_linearmask(pre_train="train").to(DEVICE)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params: {} M'.format(n_parameters / 1e6))
        model.load_state_dict(
            torch.load(config.output_dir + f"mask_unmask_model_{100}.pth", map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            model.plot = True
            # for _, ecg in enumerate(dataloader_test):
            #     for i in range(1):
            #         sig = ecg[i].reshape(1, 1280).type(torch.FloatTensor).to(DEVICE)
            #         patches, pred_pixel_values, masked_indices = model(sig)
            #         plot_maskECG(ecg[i], patches, pred_pixel_values, masked_indices)
            #     break
            dataset_dir = "/media/lzy/Elements SE/early_warning/VF_data/"
            file_data = np.load(dataset_dir + "VF_all_data.npy", allow_pickle=True).item()["trainData"]
            for j, file_name in enumerate(tqdm.tqdm(list(file_data.keys()))):
                data_ = np.array(file_data[file_name]["X"])

                data_ = torch.from_numpy(data_).type(torch.FloatTensor).to(DEVICE)
                for i in [1,2,-2,-1]:
                    sig = data_[i].reshape(1, 1280).type(torch.FloatTensor).to(DEVICE)
                    patches, pred_pixel_values, masked_indices = model(sig)
                    plot_maskECG(data_[i], patches, pred_pixel_values, masked_indices)
                break