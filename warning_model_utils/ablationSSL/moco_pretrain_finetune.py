import argparse

import torchvision.transforms as transforms_ecg
import ast
import copy
import h5py
import pickle
from scipy.signal import resample
import torch
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from torch import nn
from ablationSSL.mocoModel import CustomMoCo

method="moco"
class mocoConfig:
    # 信号参数
    target_sample_rate = 100
    input_signal_len = 1000
    input_signal_chls = 12
    # 训练时的batch大小
    batch_size = 256
    lr = 5e-4
    min_lr = 1e-5
    max_epoch = 100

    # 微调参数
    finetune_ckp=99
    cls_high_lr = 5e-4
    cls_low_lr = 5e-4
    cls_min_lr = 1e-5
    cls_epoch = 10
    cls_batch_size = 128

    def __init__(self, model_name="convnextv2_atto", dataSource='ptb-xl'):
        self.model_name = model_name
        self.dataSource = dataSource
        if self.dataSource == "ptb-xl":
            self.num_classes = 5
        if self.dataSource == "hospital":
            self.num_classes = 3
        if self.dataSource == "icbeb":
            self.num_classes = 11
        self.cls_folder = "/home/lzy/workspace/codeFile/MIClassification/modelResult/ablationSSL/" \
                          "moco_{}_{}/".format(dataSource, model_name)
        if not os.path.exists(self.cls_folder):
            os.makedirs(self.cls_folder)
        # 存储参数
        self.output_dir = "/home/lzy/workspace/codeFile/MIClassification/pretrainModel/{}/{}_checkpoint/".format(method, model_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)


import matplotlib.pyplot as plt
from ablationSSL.timeseries_transformations import GaussianNoise, RandomResizedCrop, ChannelResize, Negation, \
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

    with h5py.File("/media/lzy/Elements SE/MIClassification/pretrainingData.hdf5", 'r') as f:
        ecgSig = np.array(f['data'][:])
        print("Pretrain data: ", end="  ")
        print(len(ecgSig))
        trainData, testData  = train_test_split(ecgSig, test_size=0.1, random_state=111)

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
        self.dataSet = torch.from_numpy(np.array(ecgs))
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
                        help='base model', choices=["vit", "convnextv2_atto", "convnextv2_nano"])
    parser.add_argument("--finetune_ckp", type=int, default=10)
    args = parser.parse_args()
    print(args)

    flag = args.flag
    model_name = args.model_name
    exp_name = args.finetune_exp

    DEVICE = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')
    config = mocoConfig(model_name=model_name, dataSource=exp_name)
    config.finetune_ckp = args.finetune_ckp
    fine_tune_epoch = args.finetune_ckp
    # trainData, testData = getPretrainData()
    # dataloader = DataLoader(maeDataset(trainData), batch_size=256, shuffle=True)
    # for _, ecg in enumerate(dataloader):
    #     (x1, y1), (x2, y2) = ecg
    #     ecg_1 = x1[0]
    #     ecg_2 = x2[0]
    #     ECGplot_multiLeads(ecg_1)
    #     ECGplot_multiLeads(ecg_2)
    #     break
    if flag == 0:

        DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # 载入数据集
        trainData, testData = getPretrainData()


        model = CustomMoCo(encoder=config.model_name).to(DEVICE)
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
                moco_loss = model(data_source, DEVICE)

                loss = moco_loss

                loss.backward()
                optimizer.step()
                # global_step = len_dataloader * (epoch - 1) + i + 1
                # max_step = len_dataloader * config.max_epoch
                # model.on_train_batch_end(global_step, max_step)
                total_loss_train += loss.item()

                loop.set_description(f'Epoch [{epoch}/{config.max_epoch}]')
                loop.set_postfix(moco_loss=moco_loss.item())
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

    elif flag == 1:

        class GetLoader(torch.utils.data.Dataset):
            def __init__(self, train_flag=0, data_source="ptb-xl"):
                if data_source == "hospital":
                    self.dataPath = '/home/lzy/workspace/codeFile/MIClassification/data/acuteMI/'
                    X = []
                    if train_flag == 0:
                        with open(self.dataPath + 'train.pkl', 'rb') as f:
                            train_data = pickle.load(f)
                    elif train_flag == 1:
                        with open(self.dataPath + 'val.pkl', 'rb') as f:
                            train_data = pickle.load(f)
                    elif train_flag == 2:
                        with open(self.dataPath + 'test.pkl', 'rb') as f:
                            train_data = pickle.load(f)
                    y_train = train_data["y"]
                    y_train = y_train
                    X_train = train_data["x"].astype(np.float64)
                    aux_train = train_data["aux"]
                else:
                    self.dataPath = '/home/lzy/workspace/codeFile/MIClassification/PTB-XL/superdiagnostic/data/'
                    X = []

                    if train_flag == 0:
                        with open(self.dataPath + 'train.pkl', 'rb') as f:
                            train_data = pickle.load(f)
                    elif train_flag == 1:
                        with open(self.dataPath + 'val.pkl', 'rb') as f:
                            train_data = pickle.load(f)
                    elif train_flag == 2:
                        with open(self.dataPath + 'test.pkl', 'rb') as f:
                            train_data = pickle.load(f)

                    y_train = train_data["y"]
                    y_train = y_train.astype(np.float64)
                    X_train = train_data["x"].astype(np.float64)
                    x = []
                    for ecg in X_train:
                        x.append(self.resample_array(ecg.T))
                    X_train = np.array(x)
                    print(X_train.shape)
                    assert X_train.shape[-1] == 1000
                    aux_train = train_data["aux"]

                self.dataset = torch.from_numpy(np.array(X_train))
                self.labels = torch.from_numpy(y_train)
                self.auxInfo = aux_train
                self.n_data = len(X_train)

            def resample_array(self, original_array, new_size=1000):
                # 假设 original_array 是一个 numpy 数组，形状为 (12, 5000)
                original_size = original_array.shape[1]  # 5000

                # 创建一个新的数组，用于存储重采样结果
                resampled_array = np.zeros((original_array.shape[0], new_size))

                for i in range(original_array.shape[0]):
                    resampled_array[i, :] = resample(original_array[i, :], new_size)

                return resampled_array

            def __getitem__(self, item):

                return self.dataset[item], self.labels[item]

            def __len__(self):
                return self.n_data


        def train(ep, model, dataload):
            model.train()
            total_loss = 0
            mae_loss = 0
            n_entries = 0
            train_desc = "Epoch {:2d}: train - Loss: {:.6f}, -mae_loss: {:.6f}"
            train_bar = tqdm(initial=0, leave=True, total=len(dataload),
                             desc=train_desc.format(ep, 0, 0))
            # loss_class = nn.CrossEntropyLoss()
            loss_class = nn.BCEWithLogitsLoss()
            for data, label in dataload:
                # traces = traces.transpose(1, 2)
                data, label = data.float().to(device), label.to(device)
                # Reinitialize grad
                optimizer.zero_grad()

                # classification Forward pass
                pred = model.classify(data)
                loss = loss_class(pred, label)
                #
                # Backward pass
                loss.backward()
                # Optimize
                optimizer.step()
                # Update
                bs = len(label)
                total_loss += loss.detach().cpu().numpy()

                n_entries += bs
                # Update train bar
                train_bar.desc = train_desc.format(ep, total_loss / n_entries, mae_loss / n_entries)
                train_bar.update(1)
            train_bar.close()
            return total_loss / n_entries


        def _test_(model, dataload):
            model.eval()
            tru = []
            preds = []
            for data, label in dataload:
                # traces = traces.transpose(1, 2)
                data, label = data.float().to(device), label.to(device)
                with torch.no_grad():
                    # Reinitialize grad
                    model.zero_grad()
                    # Send to device
                    # Forward pass
                    pred = model.classify(data)

                    # _, pred = torch.max(pred, 1)
                    truth = label
                    pred = pred.cpu()
                    # acc += (pred.numpy() == truth.cpu().numpy()).sum()
                    # c += len(pred)
                    preds.append(pred.tolist())
                    tru.append(truth.cpu().tolist())

            return preds, tru


        device = torch.device('cuda:1')

        model = CustomMoCo(encoder=config.model_name).to(device)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('number of params: {} M'.format(n_parameters / 1e6))
        pretrain_param = torch.load(config.output_dir + f"{method}_pretrain_{fine_tune_epoch}.pth", map_location=device)
        model.load_state_dict(
            pretrain_param, strict=False)

        untrained_params = list(model.encoder_q.multi_label_classifier.parameters())
        untrained_param_ids = set(id(p) for p in untrained_params)
        other_params = [p for p in model.parameters() if id(p) not in untrained_param_ids]

        # 为不同参数组设置不同的学习率
        optimizer = torch.optim.AdamW([
            {'params': untrained_params, 'lr': config.cls_high_lr},
            {'params': other_params, 'lr': config.cls_low_lr}
        ], weight_decay=1e-6)
        # 为不同参数组设置不同的学习率


        # 为不同参数组设置不同的学习率
        # optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.cls_epoch, eta_min=config.cls_min_lr)

        best_model = copy.deepcopy(model)
        best_model = best_model.to(device)
        tqdm.write("Define model...")


        tqdm.write("Define dataloder...")
        train_loader = torch.utils.data.DataLoader(dataset=GetLoader(train_flag=0, data_source=config.dataSource),
                                                   batch_size=config.cls_batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=GetLoader(train_flag=1, data_source=config.dataSource),
                                                   batch_size=config.cls_batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=GetLoader(train_flag=2,  data_source=config.dataSource),
                                                   batch_size=config.cls_batch_size, shuffle=True)

        tqdm.write("Done!")

        tqdm.write("Training...")
        tqdm.write("Training...")
        start_epoch = 0
        best_loss = - np.Inf
        history = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'lr',
                                        'weighted_rmse', 'weighted_mae', 'rmse', 'mse'])
        # 可视化loss
        time_line = []
        val_acc = []
        val_loss_line = []
        plt.figure(figsize=(4, 4), dpi=200)
        for ep in range(start_epoch, config.cls_epoch):
            train_loss = train(ep, model, train_loader)
            preds111, truth = _test_(model, val_loader)
            preds_flat = [item for pre in preds111 for item in pre]
            truth_flat = [item for tru in truth for item in tru]
            acc_per_task = roc_auc_score(truth_flat, preds_flat, average='macro')
            # Save best model
            if acc_per_task > best_loss:
                best_model.load_state_dict(model.state_dict())
                # Update best validation loss
                best_loss = acc_per_task
            # Get learning rate
            for param_group in optimizer.param_groups:
                learning_rate = param_group["lr"]
            # Interrupt for minimum learning rate

            # Print message
            tqdm.write('Epoch {:2d}: \tTrain Loss {:.6f} ' \
                       '\tValid Loss {:.6f} \tLearning Rate {:.7f}\t'
                       .format(ep, train_loss, acc_per_task, learning_rate))
            # Update learning rate
            scheduler.step()
            time_line.append(ep)
            val_acc.append(acc_per_task)

        tqdm.write("training Done!")
        preds = []
        truths = []
        preds111, truth = _test_(best_model, test_loader)
        preds.append(preds111)
        truths.append(truth)
        preds_flat = [item for pred in preds for sublist in pred for item in sublist]
        truth_flat = [item for truth in truths for sublist in truth for item in sublist]
        print('\nEvaluating....')
        print(roc_auc_score(truth_flat, preds_flat, average='macro'))
