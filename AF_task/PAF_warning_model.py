import pickle

import os
import random

from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

from warning_model_utils.pretrain_model import MAE_linearmask
from warning_model_utils.warning_indicator_model import indicator_cls_model
import seaborn as sns

DEVICE =  torch.device('cpu')
dataset_dir = "./AF_data/"
paf_model_save_path = "./AF_model/"


def seed_torch(seed=1029):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 划分训练，验证
def GenerateTrainAndTest():
    with open(dataset_dir + "train_test_spilt.pkl", 'rb') as file:
        patient = pickle.load(file)
    return patient["trainPatient"], patient["valPatient"], patient["testPatient"]


def getIndicatorModel():

    model_cls = indicator_cls_model(num_classes=3).to(DEVICE)

    pretrain_weight = torch.load(paf_model_save_path + f"trs_status_dis_model.pth",
                                 map_location=DEVICE)
    model_cls.load_state_dict(pretrain_weight, strict=True)
    model_cls.eval()

    # 状态模型
    model_status = MAE_linearmask(pre_train="train").to(DEVICE)
    model_status.load_state_dict(
        torch.load("./AF_model/"
                   + f"mask_unmask_model_{100}.pth", map_location=DEVICE))
    model_status.eval()

    return model_cls, model_status


def ECG_status_Feature_extraction():
    file_list = []
    trainPatient, valPatient, testPatient = GenerateTrainAndTest()
    file_list.extend(trainPatient)
    file_list.extend(valPatient)
    file_list.extend(testPatient)

    # load model
    model_cls, model_status = getIndicatorModel()

    print(model_cls.class_center)

    indicator_data = {}
    status_data = {}
    for file_path in tqdm(file_list):
        with open(dataset_dir + file_path, 'rb') as file:
            file_data = pickle.load(file)
            data_ = file_data["X"]
            label_ = np.array(file_data["Y"]).reshape(-1, 1)
            indicator_patient = []
            data_ = torch.from_numpy(data_).type(torch.FloatTensor).to(DEVICE)
            with torch.no_grad():
                for ecg in data_:
                    ecg = ecg.unsqueeze(0)
                    [dis_0, dis_1, dis_2] = model_cls.getStatusDistance(ecg)
                    ind_status_dis_0 = torch.log(dis_0).cpu().numpy()
                    ind_status_dis_1 = torch.log(dis_1).cpu().numpy()
                    ind_status_dis_2 = torch.log(dis_2).cpu().numpy()
                    indicator_patient.append([ind_status_dis_0, ind_status_dis_1, ind_status_dis_2])
                status_feature = model_status.getEmbeding(data_)

            status_data[file_path] = status_feature
            indicator_data[file_path] = [np.array(indicator_patient), label_]

    data_save_path = "./AF_data/"

    np.save(data_save_path + "status_dis.npy", indicator_data)
    np.save(data_save_path + "status.npy", status_data)



def CausalFilter(index_list, win_size = 10):

    if len(index_list) < win_size:
        return np.mean(index_list)
    else:
        return np.mean(index_list[-win_size:])







# data loader for warning indicator
class warningDataset(torch.utils.data.Dataset):
    def __init__(self, file_list=None, task="trainData", win_size=90, dataset_type=0):

        print("data loading start!")
        self.ind_all = []
        self.label_all = []
        self.status_all = []
        self.task = task
        self.dataset_type = dataset_type
        self.data_status = "train"
        status_data = np.load("/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/status.npy", allow_pickle=True).item()
        status_dis_data = np.load("/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/status_dis.npy",
                              allow_pickle=True).item()

        count = 0

        for file_path in tqdm(file_list):

            data_status = status_data[file_path].cpu().numpy()
            data_status_dis = status_dis_data[file_path][0]
            label_ori = status_dis_data[file_path][1].reshape(-1)
            status_patient = []
            status_dis_patient = []
            label_patient = []
            for sta, sta_dis, la, in zip(data_status, data_status_dis, label_ori):
                status_patient.append(sta)
                status_dis_patient.append(sta_dis)
                label_patient.append(la)
            self.status_all.append(np.array(status_patient))
            self.ind_all.append(np.array(status_dis_patient))
            self.label_all.append(np.array(label_patient))
        self.dataSet = []
        self.status = []
        self.label = []
        self.updateObservationLength(win_size)


    def generateLabel(self, label, start, end_ind):
        # 预警标签 低风险、中风险、高风险
        if max(label) == 0:
            return [1., 0., 0.]
        else:
            pre_status = np.where(label == 1)[0][0]
            if label[end_ind] == 0:
                ratio = (pre_status - end_ind) / pre_status
                if ratio > 0.75:
                    return [0.9,  0.1, 0.]
                else:
                    return [0.5,  0.5, 0.0]
            else:
                ratio = (end_ind - pre_status) / (end_ind - start)
                if ratio < 1:
                    return [0., 0.9, 0.1]
                elif ratio > 2:
                    return [0., 0.1, .9]
                else:
                    return [0., 0.5, 0.5]


    def spiltWindowData(self, sequence, label, status, win_size=60):
        start = 0
        data_len = len(sequence)
        flag = False
        if max(label) < 1 or self.task == "testData":
            stride = 9
        else:
            stride = 6
            flag = True
        seq_seg = []
        label_seg = []
        status_seg = []

        assert len(sequence) == len(label) and len(sequence) == len(status)
        while start + win_size < data_len:
            seq_seg.append(sequence[start: start + win_size])
            label_seg.append(self.generateLabel(label, start, start + win_size))

            status_seg.append(status[start: start + win_size])
            if flag:
                pre_status = np.where(label == 2)[0][0]
                if start + win_size >= pre_status and self.task == "trainData":
                    stride = 6
            start += stride
        return np.array(seq_seg), np.array(label_seg), np.array(status_seg)

    def updateObservationLength(self, observationLength):
        del self.dataSet, self.status, self.label
        initial_len = 15000
        ind = np.empty((initial_len, observationLength, 3))
        if self.task == "trainData":
            label = np.empty((initial_len, 3))
        else:
            label = np.empty((initial_len, 1))
        status = np.empty((initial_len, observationLength, 20, 256))
        count = 0
        for data_status, seq, porb in tqdm(zip(self.status_all, self.ind_all, self.label_all)):
            seq_seg, label_seg, status_seg = self.spiltWindowData(seq, porb, data_status, win_size=observationLength)
            seq_seg = seq_seg.reshape(-1, observationLength, 3)
            num_seg = len(seq_seg)
            ind[count: count + num_seg, :, :] = seq_seg
            label[count: count + num_seg, :] = label_seg
            status[count: count + num_seg, :, :, :] = status_seg
            count += num_seg
            assert count < initial_len
        print("什么情况")
        print(count)
        print(np.sum(label,axis=0))
        if self.dataset_type == 1:
            ind = ind[:count]
            status = status[:count]
            label = label[:count]
            # 生成索引数组
            indices = np.arange(count)
            # 划分索引，而不是直接划分数据
            train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)

            self.dataSet = ind
            self.status = status
            self.label = label

            self.indices_test = test_indices
            self.indices_train = train_indices

            self.n_data = len(train_indices)
        else:
            self.dataSet = ind[:count]
            self.status = status[:count]
            self.label = label[:count]
            self.n_data = len(self.dataSet)

    def set_data_status(self, status):
        self.data_status = status
        if status == "val":
            self.n_data = len(self.indices_test)
        else:
            self.n_data = len(self.indices_train)

    def __getitem__(self, item):
        if self.data_status == "val":

            d = torch.from_numpy(np.array(self.dataSet[self.indices_test[item]]))
            s = torch.from_numpy(np.array(self.status[self.indices_test[item]]))
            l = torch.from_numpy(np.array(self.label[self.indices_test[item]]))
        else:
            d = torch.from_numpy(np.array(self.dataSet[self.indices_train[item]]))
            s = torch.from_numpy(np.array(self.status[self.indices_train[item]]))
            l = torch.from_numpy(np.array(self.label[self.indices_train[item]]))
        return d, l, s

    def __len__(self):
        return self.n_data

class warningLSTM(nn.Module):
    def __init__(self, fusion_dim=40):
        super(warningLSTM, self).__init__()
        # index 1 model
        self.status_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

        # index 2 model
        self.ind_model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        self.hidden_dim = 64
        self.num_layers = 4
        # First layers
        self.lstm1 = nn.LSTM(input_size=fusion_dim, hidden_size=self.hidden_dim, batch_first=True, bidirectional=False, num_layers=self.num_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 3),

        )

    def soft_ce_loss(self, pred, tar):
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -(tar * log_probs).sum(dim=-1).mean()
        return loss

    def forward(self, x, status, hidden=None):
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        else:
            h0 = hidden[0]
            c0 = hidden[1]
        status = self.status_model(status)
        status = torch.mean(status.permute(0, 1, 3, 2), dim=3, keepdim=True)
        status = status.squeeze(-1)

        x = self.ind_model(x[:, :, :])
        x = torch.cat([x, status], dim=2)

        lstm_output, hidden = self.lstm1(x, (h0, c0))
        # latentFeature = self.avgpool(lstm_output.permute(0, 2, 1))
        latentFeature = lstm_output[:, -1, :]
        latentFeature = latentFeature.view(latentFeature.size(0), -1)
        x = self.fc(latentFeature)
        return x, hidden


def training_one_epoch_model(model, dataloader, optimizer, epoch, max_epoch):
    loop = tqdm(enumerate(dataloader), desc="Training")
    loop.set_description(f'Epoch [{epoch}/{max_epoch}]')
    epoch_loss = 0
    model.train()
    for i, data_source in loop:
        optimizer.zero_grad()
        ecg, label = data_source[0].type(torch.FloatTensor).to(DEVICE), data_source[1].type(torch.FloatTensor).to(
            DEVICE)
        status = data_source[2].type(torch.FloatTensor).to(DEVICE)
        out_put, _ = model(ecg, status)

        loss = model.soft_ce_loss(out_put, label)
        loss.backward()
        loop.set_postfix(loss=loss.item())
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)
def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for data_source in dataloader:
            ecg, label = data_source[0].type(torch.FloatTensor).to(DEVICE), data_source[1].type(torch.FloatTensor).to(
                DEVICE)
            status = data_source[2].type(torch.FloatTensor).to(DEVICE)
            out_put, _ = model(ecg, status)
            loss = model.soft_ce_loss(out_put, label)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def getTpScore(Tp, mu=10, sigma=5):
    score_list = []
    for s in Tp:
        if s == 0:
            score_list.append(0)
        else:
            score_list.append(np.exp(-((s - mu) ** 2) / (2 * sigma ** 2)))
    return score_list

def findOrTestThreshold(pred_val, true_val, onset_time, threshold_value=None):

    acc_list = []
    forahead_time = []
    if threshold_value is None:
        threshold_list = np.arange(0.00, 1., 0.02)
    else:
        threshold_list = [threshold_value]
    for threshold in threshold_list:
        label_pred = []
        pred_time = []
        for pred_patient, onset_time_patient in zip(pred_val, onset_time):
            AF_flag = False
            for i in range(len(pred_patient)):
                if pred_patient[i] >= threshold:
                    pred_time.append(max(onset_time_patient - i, 0))
                    AF_flag = True
                    label_pred.append(1)
                    break
            if not AF_flag:
                pred_time.append(-1)
                label_pred.append(0)

        acc_list.append(accuracy_score(true_val, label_pred))
        # 预测正确的提前时间
        AF_idx = []
        for kk in range(len(true_val)):
            if true_val[kk] == 1 and label_pred[kk] == 1:
                AF_idx.append(kk)
        if len(AF_idx) == 0:
            forahead_time.append(0)
        else:
            forahead_time.append(np.mean(np.array(pred_time)[AF_idx]))
            if threshold_value is not None:
                score = np.mean(getTpScore(np.array(pred_time)[AF_idx] // 12))
                f_2 = (acc_list[0] * score * 5) / (acc_list[0] + 4 * score)
                print("HM：{}".format(f_2))

    # 创建图表和第一个y轴
    if threshold_value is None:
        x = np.arange(0.05, 1., 0.02)
        y1 = forahead_time
        y2 = acc_list
        print(max(y2))
        # plt.rcParams["font.family"] = "Calibri"
        fig, ax = plt.subplots(figsize=(4, 2), dpi=200)
        ax.set_xticks(np.arange(0, 1.2, 0.25))
        ax.set_yticks(np.arange(-0.2, 1.2, 0.2))
        ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
        ax.set(xlabel='threshold')
        ax.set(ylabel='F')
        ratio = np.array(y1) / 12 / 30
        f_2 = (np.array(y2) * ratio * 5) / (np.array(y2) + 4 * ratio)
        ax.plot(x, f_2, label="F", color="blue", alpha=0.9, lw=1.0)
        ax.plot(x[[np.argmax(f_2), np.argmax(f_2)]], [-0.2, max(f_2)], "r--")
        print(x[ np.argmax(f_2)])
        ax.legend(loc='upper right', prop={'size': 10})
        ax.set_xlim(left=0.1, right=1.)
        ax.set_ylim(top=1.15, bottom=-0.1)
        plt.show()
        return x[np.argmax(f_2)]
    else:
        f, ax = plt.subplots()
        cm = confusion_matrix(y_true=true_val, y_pred=label_pred)
        sns.heatmap(cm, annot=True, ax=ax)  # 画热力图

        ax.set_title('warning confusion matrix')  # 标题
        ax.set_xlabel('predict')  # x 轴
        ax.set_ylabel('true')  # y 轴
        # 命令行输出 混淆矩阵
        print('\nEvaluating....')
        print("TEST ACC:", accuracy_score(true_val, label_pred))
        print(classification_report(true_val, label_pred))
        print("Confusion Matrix:")
        print(cm)
        plt.show()

        return acc_list[0], forahead_time[0]

def getWarningRisk(model, status_dis, status, label, file_name = None):

    risk_value = []
    label_true = []
    status_feature = torch.from_numpy(status).unsqueeze(0).to(DEVICE)
    status_dis = torch.from_numpy(status_dis).unsqueeze(0).to(DEVICE)
    initial_len = 11
    label = label
    data_pred = []
    data_label = []

    with torch.no_grad():
        pred_0, hidden = model(status_dis[:, :initial_len, :], status_feature[:, :initial_len, :, :])

        pred = nn.Softmax()(pred_0).cpu().numpy().reshape(3)
        data_pred.append(pred)
        data_label.append(label[initial_len-1])
        r_v = 0.3 * pred[1] + 0.7 * pred[2]

        risk_value.append(r_v)
        label_true.append(label[initial_len-1])
        for i in range(initial_len, len(label)):
            pred, hidden = model(status_dis[:, i, :].unsqueeze(1), status_feature[:, i, :, :].unsqueeze(1),
                                 hidden=hidden)
            pred = nn.Softmax()(pred).cpu().numpy().reshape(3)
            data_pred.append(pred)
            data_label.append(label[i])
            r_v = 0.3 * pred[1] + 0.7 * pred[2]
            risk_value.append(r_v)
            label_true.append(label[i])

    warning_curve_save_path = "./AF_data/"
    if not os.path.exists(warning_curve_save_path):
        os.makedirs(warning_curve_save_path)

    if file_name is not None:
        np.save(warning_curve_save_path + "{}.npy".format(file_name),
                {"pred": data_pred, "label": data_label})
    return risk_value, label_true


def saveWarningCurve(model):
    model.eval()
    trainData, valData, testData = GenerateTrainAndTest()
    trainData.extend(valData)

    print(" test patient", end=" ")
    print(testData)
    status_data = np.load("/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/status.npy",
                          allow_pickle=True).item()
    status_dis_data = np.load(
        "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/status_dis.npy",
        allow_pickle=True).item()
    for j, train_patient in enumerate(trainData):
        data_status = status_data[train_patient].cpu().numpy()
        data_status_dis = np.array(status_dis_data[train_patient][0]).reshape(-1, 3)
        label_ori = status_dis_data[train_patient][1]
        status_patient = []
        status_dis_patient = []
        label_patient = []
        for sta, sta_dis, la, q in zip(data_status, data_status_dis, label_ori):
            status_patient.append(sta)
            status_dis_patient.append(sta_dis)

            label_patient.append(la)
        getWarningRisk(model, np.array(status_dis_patient),
                                                np.array(status_patient), np.array(label_patient), file_name=train_patient)

    for j, test_patient in enumerate(testData):
        data_status = status_data[test_patient].cpu().numpy()
        data_status_dis = np.array(status_dis_data[test_patient][0]).reshape(-1, 3)
        label_ori = status_dis_data[test_patient][1]
        status_patient = []
        status_dis_patient = []
        label_patient = []
        for sta, sta_dis, la in zip(data_status, data_status_dis, label_ori):

            status_patient.append(sta)
            status_dis_patient.append(sta_dis)
            label_patient.append(la)

        getWarningRisk(model, np.array(status_dis_patient), np.array(status_patient),
                                                np.array(label_patient), file_name=test_patient)



def data_visualization(indicator, label, title=None):
    fig, ax = plt.subplots(3, 1, figsize=(4, 4), dpi=200)
    ax[0].set_title(title)
    ax[0].set(xlabel='time (min)')
    ax[0].set(ylabel='indicator')
    ax[0].plot(np.arange(0, len(indicator)), indicator[:, 0], label="clu", alpha=0.99, lw=0.5)
    ax[0].legend()
    # ax[0].set_xlim(left=0, right=10.5)
    ax[0].set_ylim(top=0.1, bottom=0)
    ax[1].set(xlabel='time (min)')
    ax[1].set(ylabel='indicator')
    ax[1].plot(np.arange(0, len(indicator)), indicator[:, 1], label="cls", alpha=0.99, lw=0.5)
    ax[1].legend()
    ax[1].set_ylim(top=1.1, bottom=0)
    ax[2].set(xlabel='time (min)')
    ax[2].set(ylabel='label')
    ax[2].plot(np.arange(0, len(indicator)), label, label="label", alpha=0.99, lw=0.5)
    ax[2].legend(loc='lower left', prop={'size': 10})
    # ax[0].set_xlim(left=0, right=10.5)
    ax[2].set_ylim(top=3, bottom=0)

    plt.show()



if __name__ == "__main__":
    # run this code only once
    # ECG_status_Feature_extraction()
    paf_model_save_path = "./AF_model/"
    finetune_epoch = 50
    flag = 2 # 0 warning training  2 testing 3 case study

    if flag == 0:
        trainData, valData, testData = GenerateTrainAndTest()
        dataloader = DataLoader(warningDataset(file_list=trainData, dataset_type=1), batch_size=128, shuffle=True)

        model = warningLSTM(fusion_dim=40).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epoch, eta_min=1e-5)

        val_loss = []
        train_loss = []
        val_loss_current = 1000000
        for epoch in range(1, finetune_epoch + 1):

            epoch_loss = training_one_epoch_model(model=model, dataloader=dataloader,
                                     optimizer=optimizer, epoch=epoch,
                                     max_epoch=finetune_epoch)
            scheduler.step()
            dataloader.dataset.set_data_status("val")
            total_loss = evaluate_model(model, dataloader)
            dataloader.dataset.set_data_status("train")
            train_loss.append(epoch_loss)
            val_loss.append(total_loss)
            if total_loss < val_loss_current:
                val_loss_current = total_loss
                print('best_performance: {:.4f}'.format(total_loss))
                torch.save(model.state_dict(),
                           paf_model_save_path + "warning_model.pth".format(epoch))


        fig, ax = plt.subplots(figsize=(4, 2), dpi=200)
        ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
        ax.set(xlabel='time(min)')
        ax.set(ylabel='warning value')
        ax.plot( train_loss, label="train_loss", color="b", alpha=0.8, lw=0.6)
        ax.plot( val_loss, color="r",
                alpha=0.8, lw=0.5)
        ax.legend(loc='upper right', prop={'size': 7})

        plt.show()

    elif flag == 1:
        model = warningLSTM(fusion_dim=40).to(DEVICE)
        pretrain_weight = torch.load(
            paf_model_save_path + f"warning_model_checkpoint/warning_model.pth",
            map_location=DEVICE)
        model.load_state_dict(pretrain_weight, strict=True)
        model.eval()

        saveWarningCurve(model)

    elif flag == 2:
        # case study
        def getECGStateFeature(ecg):

            [dis_0, dis_1, dis_2] = model_cls.getStatusDistance(ecg)
            ind_status_dis_0 = torch.log(dis_0).cpu().numpy()
            ind_status_dis_1 = torch.log(dis_1).cpu().numpy()
            ind_status_dis_2 = torch.log(dis_2).cpu().numpy()
            status_feature = model_status.getEmbeding(ecg)
            return torch.tensor( np.column_stack((ind_status_dis_0, ind_status_dis_1, ind_status_dis_2))), status_feature

        def warningPlot(ecg_length, pred_curve, ecg, warning_ind, onset_time):

            ecg_no_overlap = list(ecg[0])
            for j in range(1, len(ecg)):
                ecg_no_overlap.extend(ecg[j][640:])

            fig, ax = plt.subplots(figsize=(4, 2), dpi=200)
            ax.set_xticks(np.arange(0, 40, 5))
            ax.set_yticks(np.arange(-0., 0.8, 0.3))
            ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
            ax.set(xlabel='time(min)')
            ax.set(ylabel='warning value')
            ax.plot(np.arange(10 / 12, ecg_length, 1 / 12), pred_curve, label="F", color="b", alpha=0.8, lw=0.6)
            ax.plot(np.arange(0, ecg_length + 1 / 12, 1 / 128 / 60), np.array(ecg_no_overlap) + .6, color="b", alpha=0.8, lw=0.05)

            print("预测平均值: {}".format(np.mean(pred_curve)))

            ax.plot(np.arange(10 / 12, ecg_length, 1 / 12)[warning_ind:], pred_curve[warning_ind:], color="r",
                    alpha=0.8,
                    lw=0.4)
            ax.plot(np.arange(30, ecg_length + 1 / 12, 1 / 128 / 60), np.array(ecg_no_overlap)[30 * 128 * 60:] + .6, color="r",
                    alpha=0.8, lw=0.05)
            ax.plot([onset_time / 12, onset_time / 12],
                    [-0.1, pred_curve[onset_time]], "r--", lw=0.3)

            ax.legend(loc='upper right', prop={'size': 7})
            ax.get_legend().remove()
            ax.set_xlim(left=0.1, right=ecg_length + 1)
            ax.set_ylim(top=1.65, bottom=-0.05)
            plt.show()


        # load ECG feature model
        model_cls, model_status = getIndicatorModel()
        #load warning model
        model = warningLSTM(fusion_dim=40).to(DEVICE)
        pretrain_weight = torch.load(
            paf_model_save_path + f"warning_model.pth",
            map_location=DEVICE)
        model.load_state_dict(pretrain_weight, strict=True)
        model.eval()
        initial_len = 11
        warning_idx = -1
        onset_idx = -1
        data_pred = []
        data_label = []
        risk_value = []
        ori_pred = []
        for file_name in ["PAF_record_158_4.pkl"]:
            with open(dataset_dir + file_name, 'rb') as file:
                file_data = pickle.load(file)
                ecg_data_all = file_data["X"]
                label_ = np.array(file_data["Y"]).reshape(-1, 1)
                data_ = torch.from_numpy(ecg_data_all).type(torch.FloatTensor).to(DEVICE)
                with torch.no_grad():

                    ECG_initial = data_[:initial_len, :]
                    ECG_index_2, ECG_index_1 = getECGStateFeature(ECG_initial)

                    pred_0, hidden = model(ECG_index_2.unsqueeze(0), ECG_index_1.unsqueeze(0))
                    pred = nn.Softmax()(pred_0).cpu().numpy().reshape(3)
                    data_pred.append(pred)
                    data_label.append(label_[initial_len - 1])
                    r_v = 0.3 * pred[1] + 0.7 * pred[2]
                    risk_value.append(r_v)
                    for i in range(0, len(label_) - 11):
                        ECG_index_2, ECG_index_1 = getECGStateFeature(data_[i + 11].unsqueeze(0))
                        pred, hidden = model(ECG_index_2.unsqueeze(0), ECG_index_1.unsqueeze(0),
                                             hidden=hidden)
                        pred = nn.Softmax()(pred).cpu().numpy().reshape(3)
                        data_pred.append(pred)
                        data_label.append(label_[i])
                        r_v = 0.3 * pred[1] + 0.7 * pred[2]

                        if label_[i] == 2 and onset_idx < 0:
                            onset_idx = i
                        ori_pred.append(r_v)
                        current_r_v = CausalFilter(ori_pred)
                        if current_r_v >= 0.33 and warning_idx < 0:
                            warning_idx = i
                        risk_value.append( current_r_v)

                warningPlot(len(label_) / 12, risk_value, ecg_data_all, warning_idx, onset_idx)





