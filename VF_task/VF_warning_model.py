import pickle

import os
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from warning_model_utils.pretrain_model import MAE_linearmask
from warning_model_utils.warning_indicator_model import  indicator_cls_model
import seaborn as sns

DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
dataset_dir = "/media/lzy/Elements SE/early_warning/VF_data/"
vf_model_save_path = "/media/lzy/Elements SE/early_warning/VF_model/"


def seed_torch(seed=1029):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def getIndicatorModel():

    model_name = "trs"

    model_cls = indicator_cls_model(num_classes=3).to(DEVICE)
    pretrain_weight = torch.load(
        vf_model_save_path + f"classification_model_checkpoint/trs_status_dis_model.pth",
        map_location=DEVICE)
    model_cls.load_state_dict(pretrain_weight, strict=True)
    model_cls.eval()
    # 状态模型
    model_status = MAE_linearmask(pre_train="train").to(DEVICE)
    model_status.load_state_dict(
        torch.load("/media/lzy/Elements SE/early_warning/pretrain_result/checkpoint/"
                   + f"mask_unmask_model_{100}.pth", map_location=DEVICE))
    model_status.eval()

    return model_cls, model_status


def indicator_extraction():

    # load model
    model_cls, model_status = getIndicatorModel()

    indicator_data = {}
    status_data = {}
    for dataset in ["trainData", "testData"]:
        indicator_dataset = {}
        status_dataset = {}
        file_data = np.load(dataset_dir + "VF_all_data.npy", allow_pickle=True).item()[dataset]
        for file_name in tqdm(list(file_data.keys())):
            data_ = np.array(file_data[file_name]["X"])
            label_ = np.array(file_data[file_name]["Y"]).reshape(-1, 1)
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

            status_dataset[file_name] = status_feature
            indicator_dataset[file_name] = [indicator_patient, label_]

        indicator_data[dataset] = indicator_dataset
        status_data[dataset] = status_dataset

    np.save(dataset_dir + "status_dis.npy", indicator_data)
    np.save(dataset_dir + "status.npy", status_data)




# data loader for warning indicator

class warningDataset(torch.utils.data.Dataset):
    def __init__(self, task="trainData", win_size=30, dataset_type=0):

        print("data loading start!")
        self.ind_all = []
        self.label_all = []
        self.status_all = []
        self.task = task
        self.dataset_type = dataset_type
        self.data_status = "trainData"
        status_data = np.load(dataset_dir + "status.npy", allow_pickle=True).item()
        status_dis_data = np.load(dataset_dir + "status_dis.npy",
                              allow_pickle=True).item()

        for file_path in tqdm(status_dis_data[self.data_status].keys()):

            data_status = status_data[self.data_status][file_path].cpu().numpy()
            data_status_dis = status_dis_data[self.data_status][file_path][0]
            label_ori = status_dis_data[self.data_status][file_path][1].reshape(-1)
            status_patient = []
            status_dis_patient = []
            label_patient = []
            for sta, sta_dis, la, in zip(data_status, data_status_dis, label_ori):
                status_patient.append(sta)
                status_dis_patient.append(sta_dis)

                if "VF" in file_path:
                    la = la + 1
                else:
                    la = la
                label_patient.append(la)

            self.status_all.append(np.array(status_patient))
            self.ind_all.append(np.array(status_dis_patient))

            self.label_all.append(np.array(label_patient))

        self.dataSet = []
        self.status = []
        self.label = []
        self.updateObservationLength(win_size)
        # self.dataSet = torch.from_numpy(data)
        # self.label = torch.from_numpy(label)


    def generateLabel(self, label, start, end_ind):
        # 预警标签 低风险、中风险、高风险
        if max(label) == 0:
            return [1., 0., 0.]
        else:
            pre_status = np.where(label == 2)[0][0]
            if label[end_ind] == 1:
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



    def spiltWindowData(self, sequence, label, status, win_size=224):
        start = 0
        data_len = len(sequence)
        flag = False
        if max(label) < 3 or self.task == "testData":
            stride = win_size // 6
        else:
            stride = 2
            flag = True
        seq_seg = []
        label_seg = []
        status_seg = []

        assert len(sequence) == len(label) and len(sequence) == len(status)
        while start + win_size < data_len:
            seq_seg.append(sequence[start: start + win_size])
            if self.task == "trainData":
                label_seg.append(self.generateLabel(label, start, start + win_size))
            # else:
            #     label_seg.append(self.testLabel(label,start, start + win_size))
            status_seg.append(status[start: start + win_size])
            if flag:
                pre_status = np.where(label == 2)[0][0]
                if start + win_size >= pre_status and self.task == "trainData":
                    stride = 2
            start += stride
        return np.array(seq_seg), np.array(label_seg), np.array(status_seg)

    def updateObservationLength(self, observationLength):
        del self.dataSet, self.status, self.label
        initial_len = 5000
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
        print(f"total_number:{count}")
        # if self.dataset_type == 1:
        ind = ind[:count]
        status = status[:count]
        label = label[:count]
        # 生成索引数组
        indices = np.arange(count)
        # 划分索引，而不是直接划分数据
        train_indices, test_indices = train_test_split(indices, test_size=0.3, random_state=42)
        # 现在，根据索引划分X, Y, Z数据
        ind_train = ind[train_indices]
        ind_test = ind[test_indices]
        status_train = status[train_indices]
        status_test = status[test_indices]
        label_train = label[train_indices]
        label_test = label[test_indices]
        del status, ind, label
        self.dataSet = ind_train
        self.status = status_train
        self.label = label_train
        self.dataSet_test = ind_test
        self.status_test = status_test
        self.label_test = label_test
        self.n_data = len(self.dataSet)
        # else:
        #     self.dataSet = ind[:count]
        #     self.status = status[:count]
        #     self.label = label[:count]
        #     self.n_data = len(self.dataSet)

    def set_data_status(self, status):
        self.data_status = status
        if status == "val":
            self.n_data = len(self.dataSet_test)
        else:
            self.n_data = len(self.dataSet)

    def __getitem__(self, item):
        if self.data_status == "val":
            d = torch.from_numpy(np.array(self.dataSet_test[item]))
            s = torch.from_numpy(np.array(self.status_test[item]))
            l = torch.from_numpy(np.array(self.label_test[item]))
        else:
            d = torch.from_numpy(np.array(self.dataSet[item]))
            s = torch.from_numpy(np.array(self.status[item]))
            l = torch.from_numpy(np.array(self.label[item]))
        return d, l, s

    def __len__(self):
        return self.n_data

class warningLSTM(nn.Module):
    def __init__(self, fusion_dim=40):
        super(warningLSTM, self).__init__()
        # status model
        self.status_model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )

        # ind model
        self.ind_model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

        self.hidden_dim = 64
        self.num_layers = 4
        # First layers
        self.fusion_dim = fusion_dim
        self.lstm1 = nn.LSTM(input_size=fusion_dim, hidden_size=self.hidden_dim, batch_first=True, bidirectional=False, num_layers=self.num_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 3),

        )

    def soft_ce_loss(self, pred, tar):
        # 计算对数概率
        log_probs = F.log_softmax(pred, dim=-1)
        # 计算软标签的交叉熵损失
        # 这里用到的是对数概率和软标签直接的点积求和的负值
        loss = -(tar * log_probs).sum(dim=-1).mean()
        return loss

    def forward(self, x, status, hidden=None):
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 乘以2因为是双向
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
def evaluate_model(model, dataloader, show=False):
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


def findOrTestThreshold(pred_val, true_val, onset_time, threshold_value=None):

    acc_list = []
    forahead_time = []
    if threshold_value is None:
        threshold_list = np.arange(0.05, 1., 0.02)
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

        # plt.fill_between(range(len(test_X)), pred_y + uncertainty, pred_y - uncertainty, alpha=0.1)
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
        ratio = np.array(y1) / 12 / 10
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
    end_id = len(label)
    # end_ids = np.where(label==4)[0]
    # if len(end_ids) == 0:
    #     end_id = len(label) - 1
    # else:
    #     end_id = end_ids[0]

    risk_value = []
    label_true = []
    status_feature = torch.from_numpy(status[:end_id]).unsqueeze(0).to(DEVICE)
    status_dis = torch.from_numpy(status_dis[:end_id]).unsqueeze(0).to(DEVICE)
    initial_len = 5
    label = label[:end_id]

    data_pred = []
    data_label = []

    with torch.no_grad():
        pred_0, hidden = model(status_dis[:, :initial_len, :], status_feature[:, :initial_len, :, :])

        pred = nn.Softmax()(pred_0).cpu().numpy().reshape(3)
        data_pred.append(pred)
        data_label.append(label[initial_len-1])
        r_v =  0.3 * pred[1] + 0.7 * pred[2]


        risk_value.append(r_v )
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
    if file_name is not None:
        np.save(dataset_dir + "warning_risk_value/{}.npy".format(file_name),
                {"pred": data_pred, "label": data_label})
    return risk_value, label_true



def testWarning(model):
    model.eval()
    status_data = np.load(dataset_dir + "status.npy",
                          allow_pickle=True).item()
    status_dis_data = np.load(dataset_dir + "status_dis.npy",
        allow_pickle=True).item()

    # get threshold
    pred_val = []
    true_val = []
    onset_time = []
    for j, train_patient in enumerate(status_dis_data["trainData"].keys()):

        data_status = status_data["trainData"][train_patient].cpu().numpy()
        data_status_dis = np.array(status_dis_data["trainData"][train_patient][0]).reshape(-1, 3)
        # quality = status_dis_data["trainData"][train_patient][2]
        label_ori = status_dis_data["trainData"][train_patient][1]
        status_patient = []
        status_dis_patient = []
        label_patient = []
        for sta, sta_dis, la in zip(data_status, data_status_dis, label_ori):

                status_patient.append(sta)
                status_dis_patient.append(sta_dis)

                if "VF" in train_patient:
                    la = la + 1
                else:
                    la = la
                label_patient.append(la)


        pred_curve, true_curve = getWarningRisk(model, np.array(status_dis_patient), np.array(status_patient), np.array(label_patient)
                                                ,file_name=train_patient)
        pred_val.append(pred_curve)
        # true_val.append(true_curve)
        if "VF" in train_patient:
            true_val.append(1)
            ind = np.where(true_curve==3)[0]
            if len(ind) == 0:
                onset_time.append(len(true_curve))
            else:
                onset_time.append(ind[0])
        else:
            true_val.append(0)
            onset_time.append(-1)
    threshold = findOrTestThreshold(pred_val, true_val, onset_time)


    # test
    pred_val = []
    true_val = []
    onset_time = []
    for j, test_patient in enumerate(status_dis_data["testData"].keys()):
        data_status = status_data["testData"][test_patient].cpu().numpy()
        data_status_dis = np.array(status_dis_data["testData"][test_patient][0]).reshape(-1, 3)
        # quality = status_dis_data["testData"][test_patient][2]
        label_ori = status_dis_data["testData"][test_patient][1]
        status_patient = []
        status_dis_patient = []
        label_patient = []
        for sta, sta_dis, la, in zip(data_status, data_status_dis, label_ori):
            # if q == 0:
                status_patient.append(sta)
                status_dis_patient.append(sta_dis)
                if "non" in test_patient:
                    la = la + 1
                elif "VF" in test_patient:
                    la = la + 1
                else:
                    la = la
                label_patient.append(la)

        pred_curve, true_curve = getWarningRisk(model, np.array(status_dis_patient), np.array(status_patient),
                                                np.array(label_patient), file_name=test_patient)





def data_visualization(indicator, label):
    fig, ax = plt.subplots(3, 1, figsize=(4, 4), dpi=200)
    ax[0].set(xlabel='time (min)')
    ax[0].set(ylabel='indicator')
    ax[0].plot(np.arange(0, len(indicator)), indicator[:, 0], label="clu", alpha=0.99, lw=0.5)
    ax[0].legend()
    # ax[0].set_xlim(left=0, right=10.5)
    # ax[0].set_ylim(top=3, bottom=0)
    ax[1].set(xlabel='time (min)')
    ax[1].set(ylabel='indicator')
    ax[1].plot(np.arange(0, len(indicator)), indicator[:, 1], label="cls", alpha=0.99, lw=0.5)
    ax[1].legend()
    # ax[0].set_xlim(left=0, right=10.5)
    ax[2].set(xlabel='time (min)')
    ax[2].set(ylabel='label')
    ax[2].plot(np.arange(0, len(indicator)), label, label="label", alpha=0.99, lw=0.5)
    ax[2].legend(loc='lower left', prop={'size': 10})
    # ax[0].set_xlim(left=0, right=10.5)
    ax[2].set_ylim(top=3, bottom=-1)

    plt.show()



if __name__ == "__main__":
    # 仅调用一次
    indicator_extraction()

    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    vf_model_save_path = "/media/lzy/Elements SE/early_warning/VF_model/"
    finetune_epoch = 20
    flag = 1 # 0 warning training 2 testing

    if flag == 0:

        dataloader = DataLoader(warningDataset(), batch_size=32, shuffle=True)
        # dataloader_val = DataLoader(warningDataset(), batch_size=32, shuffle=True)
        model = warningLSTM().to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epoch, eta_min=1e-5)

        val_loss = []
        train_loss = []
        val_loss_current = 1000000

        for epoch in range(1, finetune_epoch + 1):
            print("train data:{}".format(len(dataloader)))
            epoch_loss = training_one_epoch_model(model=model, dataloader=dataloader,
                                     optimizer=optimizer, epoch=epoch,
                                     max_epoch=finetune_epoch)
            scheduler.step()

            dataloader.dataset.set_data_status("val")
            print("val data:{}".format(len(dataloader)))
            total_loss = evaluate_model(model, dataloader)
            dataloader.dataset.set_data_status("trainData")
            train_loss.append(epoch_loss)
            val_loss.append(total_loss)
            if total_loss < val_loss_current:
                val_loss_current = total_loss
                save_path = vf_model_save_path + "warning_model_checkpoint/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                print('best_performance: {:.4f}'.format(total_loss))
                torch.save(model.state_dict(),
                           save_path + "warning_model.pth")


        fig, ax = plt.subplots(figsize=(4, 2), dpi=200)
        # ax.set_xticks(np.arange(0, 40, 4))
        # ax.set_yticks(np.arange(-0., 1.5, 0.4))
        ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
        ax.set(xlabel='time(min)')
        ax.set(ylabel='warning value')

        ax.plot( train_loss, label="train_loss", color="b", alpha=0.8, lw=0.6)
        ax.plot( val_loss, color="r",
                alpha=0.8, lw=0.5)

        ax.legend(loc='upper right', prop={'size': 7})

        #             plt.savefig("./图/PAFwarning_casestudy-N-n410.svg", format='svg')
        plt.show()


    elif flag == 1:
        model = warningLSTM().to(DEVICE)
        pretrain_weight = torch.load(
            vf_model_save_path + f"warning_model_checkpoint/warning_model.pth",
            map_location=DEVICE)
        model.load_state_dict(pretrain_weight, strict=True)
        model.eval()

        testWarning(model)

    elif flag==2:
        def movingAvgWindow(index_list):
            accuList = []
            win_size = 3
            for i in range(1, 1 + len(index_list)):

                if i < win_size:
                    current = np.mean(index_list[:i])
                else:
                    current = np.mean(index_list[i - win_size: i])
                accuList.append(current)
            return accuList


        def getTpScore(Tp, mu=5, sigma=2.5):
            score_list = []
            for s in Tp:
                if s == 0:
                    score_list.append(0)
                else:
                    score_list.append(np.exp(-((s - mu) ** 2) / (2 * sigma ** 2)))
            return score_list

        status_dis_data = np.load(dataset_dir + "status_dis.npy",
                                  allow_pickle=True).item()

        trainData = status_dis_data["trainData"].keys()
        testData = status_dis_data["testData"].keys()

        print(" test patient", end=" ")
        print(testData)

        # get threshold
        pred_val = []
        true_val = []
        onset_time = []
        weight = 0.3
        acc_list = []
        forahead_time = []
        for j, train_patient in enumerate(trainData):

            warning_value = np.load(dataset_dir +"warning_risk_value/{}.npy".format(train_patient),
                                    allow_pickle=True).item()
            pred_value = np.array(warning_value["pred"])
            pred_curve = weight * pred_value[:, 1] + (1-weight) * pred_value[:, 2]
            pred_curve = movingAvgWindow(pred_curve)
            true_curve = np.array(warning_value["label"])

            pred_val.append(pred_curve)
            if "VF" in train_patient:
                true_val.append(1)
                ind = np.where(true_curve == 3)[0]
                if len(ind) == 0:
                    onset_time.append(len(true_curve))
                else:
                    onset_time.append(ind[0])
            else:
                true_val.append(0)
                onset_time.append(-1)
        for threshold in np.arange(0.05, 1., 0.02):
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
            VF_idx = []
            for kk in range(len(true_val)):
                if true_val[kk] == 1 and label_pred[kk] == 1:
                    VF_idx.append(kk)
            if len(VF_idx) == 0:
                forahead_time.append(0)
            else:
                score = np.mean(getTpScore(np.array(pred_time)[VF_idx] // 12))
                forahead_time.append(score)

        x = np.arange(0.05, 1., 0.02)
        y1 = forahead_time
        y2 = acc_list


        # plt.rcParams["font.family"] = "Calibri"
        fig, ax = plt.subplots(figsize=(4, 2), dpi=200)
        ax.set_xticks(np.arange(0, 1.2, 0.25))
        ax.set_yticks(np.arange(-0.2, 1.2, 0.2))
        ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
        ax.set(xlabel='threshold')
        ax.set(ylabel='F')
        ratio = np.array(y1)
        f_2 = (np.array(y2) * ratio * 5) / (np.array(y2) + 4 * ratio)
        print(max(f_2))
        ax.plot(x, f_2, label="F", color="blue", alpha=0.9, lw=1.0)
        ax.plot(x[[np.argmax(f_2), np.argmax(f_2)]], [-0.2, max(f_2)], "r--")
        print(x[np.argmax(f_2)])
        ax.legend(loc='upper right', prop={'size': 10})
        ax.set_xlim(left=0.1, right=1.)
        ax.set_ylim(top=1.15, bottom=-0.1)
        plt.show()


        # test
        pred_val = []
        true_val = []
        onset_time = []
        for j, test_patient in enumerate(testData):

            warning_value = np.load(dataset_dir + "warning_risk_value/{}.npy".format(test_patient), allow_pickle=True).item()
            pred_value = np.array(warning_value["pred"])
            pred_curve = weight * pred_value[:, 1] + (1 - weight) * pred_value[:, 2]
            pred_curve = movingAvgWindow(pred_curve)
            true_curve = np.array( warning_value["label"])

            pred_val.append(pred_curve)
            # true_val.append(true_curve)
            if "VF" in test_patient:
                true_val.append(1)
                ind = np.where(true_curve == 3)[0]
                if len(ind) == 0:
                    onset_time.append(len(true_curve))
                else:
                    onset_time.append(ind[0])
            else:
                true_val.append(0)
                onset_time.append(-1)
            warning_time = np.where(pred_curve > x[np.argmax(f_2)])[0]

            plt.title(test_patient)
            plt.plot(range(len(true_curve)), np.array(true_curve) / 3, label="prob", c="red", marker="x")
            plt.plot(range(len(true_curve)), pred_curve, label="pred", )
            plt.plot(range(len(true_curve)), [x[np.argmax(f_2)]] * len(true_curve), 'r--', )
            if len(warning_time) > 0:
                warning_time = warning_time[0]
                plt.plot([warning_time, warning_time], [0, x[np.argmax(f_2)]],'k--', )
            plt.ylim(-0.2, 1.5)
            plt.legend()
            plt.savefig(dataset_dir + "warning_result/{}.png".format(test_patient), format='png')
            plt.show()

        print("final test:   ")
        print(findOrTestThreshold(pred_val, true_val, onset_time, threshold_value=x[np.argmax(f_2)]))






