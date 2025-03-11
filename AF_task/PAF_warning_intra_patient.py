import pickle

import os
import random
from collections import Counter

import pandas as pd
from scipy import interpolate

from scipy.stats import norm
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

from previousModel.cnnLSTM4Classification import ECGNet
from warning_model_utils.pretrain_model import MAE_linearmask
from warning_model_utils.warning_indicator_model import indicator_clu_model, indicator_cls_model, indicator_vqvae_model, \
    indicator_mae_model
import seaborn as sns

DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
dataset_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/"
paf_model_save_path = "/media/lzy/Elements SE/early_warning/PAF_model/"


def seed_torch(seed=1029):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class taskDataset(torch.utils.data.Dataset):

    def __init__(self, file_list=None, train=0):
        dataset_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/"
        data = np.empty((0, 1280))
        label = np.empty((0, 1))

        print("data loading start!")
        for file_path in tqdm(file_list):

            with open(dataset_dir + file_path, 'rb') as file:
                file_data = pickle.load(file)
                data_ = file_data["X"]
                label_ = np.array(file_data["Y"]).reshape(-1, 1)

            if "PAF" in file_path:
                data_ = np.array(data_)
                label_ = np.array(label_)
                pre_ids = np.where(label_ < 2)[0]
                pre_len = len(pre_ids)
                label_[pre_ids[-pre_len // 3:]] = 1
                # idx_normal = np.where(label_ == 0)[0]
                # label_[idx_normal] = 1
                idx_abnormal = np.where(label_ > 0)[0]
                # idx_abnormal = list(range(pre_len // 3, 2 * pre_len // 3)) + list(idx_abnormal)

                data_ = np.array(data_)[idx_abnormal]
                label_ = np.array(label_)[idx_abnormal]

            else:
                data_ = np.array(data_)[:120]
                label_ = np.array(label_)[:120]

            patient_ecg = []
            patient_label = []

            for (e, l,) in zip(data_, label_):
                patient_ecg.append(e)
                patient_label.append(l)

            data = np.concatenate((data, patient_ecg), axis=0)
            label = np.concatenate((label, patient_label), axis=0)

        label = label.reshape(-1)

        # data_train, data_test, label_train, label_test = train_test_split(data, label,
        #                                                                   test_size=0.2, random_state=42)

        self.data_status = train
        print("label dis")
        print(Counter(label))
        self.dataSet = torch.from_numpy(data)
        self.label = torch.from_numpy(label)
        # self.dataSet_test = torch.from_numpy(data_test)
        # self.label_test = torch.from_numpy(label_test)
        self.n_data = len(self.dataSet)

    def set_data_status(self, status):
        # self.data_status = status
        # if status == 1:
        #     self.n_data = len(self.dataSet_test)
        # else:
            self.n_data = len(self.dataSet)

    def __getitem__(self, item):
        # if self.data_status == 1:
        #     return self.dataSet_test[item], self.label_test[item]
        # else:
            return self.dataSet[item], self.label[item]

    def __len__(self):
        return self.n_data


def train_status_dis_model( task="inter"):
    def training_one_epoch_dis_model(model, dataloader, optimizer, epoch, max_epoch,):
        loop = tqdm(enumerate(dataloader), desc="Training")
        loop.set_description(f'Epoch [{epoch}/{max_epoch}]')
        model.train()
        epoch_loss = 0
        for i, data_source in loop:
            optimizer.zero_grad()
            ecg, label = data_source[0].type(torch.FloatTensor).to(DEVICE), data_source[1].type(torch.FloatTensor).to(
                DEVICE)
            out_put = model(ecg)
            loss = nn.CrossEntropyLoss()(out_put, label.long())
            loss.backward()
            loop.set_postfix(loss=loss.item())
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(dataloader)

    def updateClassCenter(model, dataloader, dataloader_test):
        model.eval()
        class_feature_0 = np.empty((0, 512))
        class_feature_1 = np.empty((0, 512))
        class_feature_2 = np.empty((0, 512))
        for dataloader in [dataloader, dataloader_test]:
            # dataloader.dataset.set_data_status(status)
            for data_source in dataloader:
                ecg, label = data_source[0].type(torch.FloatTensor).to(DEVICE), data_source[1]
                label = label.numpy()

                out_put = model.getEmbedding(ecg)

                idx = np.where(label == 0)[0]
                out_put_cls = out_put.detach().cpu().numpy()[idx]
                class_feature_0 = np.concatenate((class_feature_0, out_put_cls), axis=0)
                idx = np.where(label == 1)[0]
                out_put_cls = out_put.detach().cpu().numpy()[idx]
                class_feature_1 = np.concatenate((class_feature_1, out_put_cls), axis=0)
                idx = np.where(label == 2)[0]
                out_put_cls = out_put.detach().cpu().numpy()[idx]
                class_feature_2 = np.concatenate((class_feature_2, out_put_cls), axis=0)

        class_feature = {0: torch.from_numpy(class_feature_0), 1: torch.from_numpy(class_feature_1),
                         2: torch.from_numpy(class_feature_2)}
        model.update_center(class_feature)

    def evaluate_dis_model(model, dataloader):
        model.eval()
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for data_source in dataloader:
                ecg, label = data_source[0].type(torch.FloatTensor).to(DEVICE), data_source[1].type(
                    torch.FloatTensor).to(
                    DEVICE)
                out_put = model(ecg)

                _, predicted = torch.max(out_put.data, 1)
                true_labels += label.tolist()
                pred_labels += predicted.tolist()

            return f1_score(true_labels, pred_labels, average='macro')

    trainPatient, valPatient, testPatient = GenerateTrainAndTest(type=task)
    print(len(trainPatient))
    print(len(valPatient))
    print(len(testPatient))
    dataloader = DataLoader(taskDataset(file_list=trainPatient, train=2), batch_size=128, shuffle=True)
    dataloader_val = DataLoader(taskDataset(file_list=valPatient, train=2), batch_size=256, shuffle=False)
    dataloader_test = DataLoader(taskDataset(file_list=testPatient, train=2), batch_size=256, shuffle=False)
    model = indicator_cls_model(num_classes=3).to(DEVICE)
    pretrain_weight = torch.load("/media/lzy/Elements SE/early_warning/pretrain_result/checkpoint/"
                                            + f"mask_unmask_model_{100}.pth", map_location=DEVICE)
    model.load_state_dict(pretrain_weight, strict=False)

    low_lr = 1e-5  # 预训练模块的学习率
    high_lr = 1e-3  # 没有预训练权重的模块的学习率

    # 将模型中的预训练权重参数和非预训练权重参数分为两组
    pretrained_params = list(model.encoder.parameters()) + \
                        list(model.to_patch.parameters()) + list(model.patch_to_emb.parameters())
    pretrained_param_ids = set(id(p) for p in pretrained_params)
    other_params = [p for p in model.parameters() if id(p) not in pretrained_param_ids]

    # 为不同参数组设置不同的学习率
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': low_lr},
        {'params': other_params, 'lr': high_lr}
    ], weight_decay=1e-6)
    finetune_epoch = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-3)
    val_loss_current = 0

    for epoch in range(1, finetune_epoch + 1):

        training_one_epoch_dis_model(model=model, dataloader=dataloader,
                                     optimizer=optimizer, epoch=epoch,
                                     max_epoch=finetune_epoch)
        scheduler.step()

        # dataloader.dataset.set_data_status(1)
        total_loss = evaluate_dis_model(model, dataloader_val, )
        # dataloader.dataset.set_data_status(0)
        if total_loss > val_loss_current:
            val_loss_current = total_loss
            updateClassCenter(model, dataloader, dataloader_val)
            save_path = paf_model_save_path + "classification_model_checkpoint/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            print('best_performance: {:.4f}'.format(total_loss))
            torch.save(model.state_dict(),
                       save_path + "trs_status_dis_model_{}.pth".format(task))
            print("test result")
            print('best_performance: {:.4f}'.format(evaluate_dis_model(model, dataloader_test, )))




# 划分训练，验证
def GenerateTrainAndTest(type="intra"):
    dataset_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/"

    with open(dataset_dir + "train_test_spilt.pkl", 'rb') as file:
        patient = pickle.load(file)
    trainData, valData, testData = patient["trainPatient"], patient["valPatient"], patient["testPatient"]
    trainData.extend(valData)
    trainData.extend(testData)
    af_count = {}
    nsr_count = {}
    for file in trainData:
        if "record" in file and "PAF" in file:
            patient = "PAF_record_" + file.split("_")[-2]
            if patient in af_count.keys():
                af_count[patient].append(int(file.split("_")[-1].split(".")[0]))
            else:
                af_count[patient] = [int(file.split("_")[-1].split(".")[0])]
        if "NSR_" in file and "_nsr_" in file:
            patient = "NSR_" + file.split("_")[1] + "_nsr_"
            if patient in nsr_count.keys():
                nsr_count[patient].append(int(file.split("_")[-1].split(".")[0]))
            else:
                nsr_count[patient] = [int(file.split("_")[-1].split(".")[0])]

    filtered_paf = {k: v for k, v in af_count.items() if len(v) > 1}
    filtered_nsr = {k: v for k, v in nsr_count.items() if len(v) > 1}

    testPatient = []
    trainPatient = []

    if type == "intra":
        for (patient, ids) in filtered_paf.items():
            index_ordered = sorted(ids)
            for id in index_ordered[:-1]:
                trainPatient.append(patient + "_" + str(id) + ".pkl")
            testPatient.append(patient + "_" + str(index_ordered[-1]) + ".pkl")
        for (patient, ids) in filtered_nsr.items():
            index_ordered = sorted(ids)
            for id in index_ordered[:-1]:
                trainPatient.append(patient + str(id) + ".pkl")
            testPatient.append(patient + str(index_ordered[-1]) + ".pkl")
        trainPatient, valPatient = train_test_split(trainPatient, test_size=0.2, random_state=32)
    else:
        trainData, testData = train_test_split(list(filtered_paf.keys()), test_size=0.2, random_state=32)
        for patient in trainData:
            ids = filtered_paf[patient]
            for id in ids:
                trainPatient.append(patient + "_" + str(id) + ".pkl")
        for patient in testData:
            ids = filtered_paf[patient]
            for id in ids:
                testPatient.append(patient + "_" + str(id) + ".pkl")

        trainData, testData = train_test_split(list(filtered_nsr.keys()), test_size=0.2, random_state=32)

        for patient in trainData:
            ids = filtered_nsr[patient]
            for id in ids:
                trainPatient.append(patient + str(id) + ".pkl")
        for patient in testData:
            ids = filtered_nsr[patient]
            for id in ids:
                testPatient.append(patient + str(id) + ".pkl")
        trainPatient, valPatient = train_test_split(trainPatient, test_size=0.2, random_state=32)

    return trainPatient, valPatient, testPatient


def getIndicatorModel():
    model_clu = "mae"
    if model_clu == "mae":
        model_clu = indicator_mae_model().to(DEVICE)
        pretrain_weight = torch.load(paf_model_save_path + f"cluster_model_checkpoint/mae_cluster_model.pth",
                                     map_location=DEVICE)
        model_clu.load_state_dict(pretrain_weight, strict=True)
        model_clu.pre_train = "eval"
    elif model_clu == "ae":
        model_clu = indicator_clu_model().to(DEVICE)
        pretrain_weight = torch.load(paf_model_save_path + f"cluster_model_checkpoint/cluster_model.pth",
                                     map_location=DEVICE)
        model_clu.load_state_dict(pretrain_weight, strict=True)

    model_clu.eval()

    model_name = "trs"
    if model_name == "trs":
        model_cls = indicator_cls_model().to(DEVICE)
    else:
        model_cls = ECGNet().to(DEVICE)
    pretrain_weight = torch.load(
        paf_model_save_path + f"classification_model_checkpoint/{model_name}_classification_model.pth",
        map_location=DEVICE)
    model_cls.load_state_dict(pretrain_weight, strict=True)
    model_cls.eval()

    # 状态模型
    model_status = MAE_linearmask(pre_train="train").to(DEVICE)
    model_status.load_state_dict(
        torch.load("/media/lzy/Elements SE/early_warning/pretrain_result/checkpoint/"
                   + f"mask_unmask_model_{100}.pth", map_location=DEVICE))
    model_status.eval()

    return model_clu, model_cls, model_status

def getRpeak(signal,fs=128):
    '''NeuroKit2 ECG Analysis'''
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    return rpeaks["ECG_R_Peaks"]
def generateProbLabel(ecgSegs,state_label, mu=240, sigma=60, delta=120,
                            hrv_threshold=50, lambda_coef=0.1):
        """
        生成动态概率标签的完整实现

        参数：
        num_windows : int - 总窗口数（示例中为300）
        r_peaks_list : list - 每个窗口的R波时间列表（单位：秒）
        mu : int - 预警区起点（窗口索引位置）
        sigma : float - 状态过渡速度（窗口数）
        delta : int - S1持续时间（窗口数）
        hrv_threshold : float - HRV异常阈值(ms)
        lambda_coef : float - 指数修正系数
        window_duration : int - 窗口时长(秒)

        返回：
        labels : ndarray - 形状为(num_windows, 3)的概率矩阵
        """
        num_windows = len(ecgSegs)
        r_peaks_list = []
        for ecg in ecgSegs:
            r_peaks_list.append(getRpeak(ecg))



        # 初始化标签矩阵
        labels = np.zeros((num_windows, 3))
        id  = np.where(state_label == 2)[0][0]



        # 时间动态概率计算
        for t in range(num_windows):
            # 计算基础概率
            if t >= id:
                # p2 = norm.cdf(t, loc=mu, scale=sigma)
                p2 = 0.9
                p1 = 0.1
                p0 = 0.0
            elif t < 240:
                p2 = 0.0
                p1 = 0.1
                p0 = 0.9
            else:
                p2 = 0.0
                p1 = 0.9
                p0 = 0.1

            labels[t] = [p0, p1, p2]


        # HRV特征计算与概率修正
        rmssd_list =[]
        for t, r_peaks in enumerate(r_peaks_list):
            if len(r_peaks) < 2:  # 至少需要两个R波
                continue

            # 计算RR间期（单位：ms）


            # 计算时域特征
            # try:
            if t < 60:
                rr_intervals = []
                for i in range(60):
                    if len(r_peaks_list[i]) < 2:  # 至少需要两个R波
                        continue
                    rr_intervals.extend(np.diff(r_peaks_list[i]) / 128 * 1000)
            else:
                rr_intervals = []
                for i in range(t-60, t):
                    if len(r_peaks_list[i]) < 2:  # 至少需要两个R波
                        continue
                    rr_intervals.extend(np.diff(r_peaks_list[i]) / 128 * 1000)
            diffs = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(diffs ** 2))

            # except:
            #     continue
            rmssd_list.append(rmssd)
            # 特征加权（示例使用RMSSD）

        rmssd_list = np.clip(rmssd_list, 0, 300)
        # 第一步：长度标准化
        if len(rmssd_list) < len(r_peaks_list):
            # 创建插值函数（使用实际索引）
            x_old = np.linspace(0, 1, len(rmssd_list))
            x_new = np.linspace(0, 1, len(r_peaks_list))

            # 处理单点特殊情况
            if len(rmssd_list) == 1:
                return np.full(r_peaks_list, rmssd_list[0])

            # 线性插值
            f = interpolate.interp1d(x_old, rmssd_list, kind='linear')
            processed = f(x_new)
        else:
            processed = rmssd_list.copy()

        # 第二步：滑动平均（处理边界）
        # 使用对称窗口，边缘反射填充
        # pad_width = 30  # 窗口长度3需要前后各填充1个
        # padded = np.pad(processed, pad_width, mode='minimum')
        #
        # # 卷积操作实现滑动平均
        # kernel = np.ones(61) / 61
        # smoothed = np.convolve(padded, kernel, mode='valid')
        smoothed = processed
        hrv_threshold = 100
        plt.figure()
        plt.plot(smoothed, "r")
        plt.show()
        idx = np.where(smoothed > 500)[0]
        if len(idx) > 0:
            print(idx[0])
            fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
            ax.minorticks_on()
            ax.set_xticks(np.arange(0, 25, 1))
            ax.set(xlabel='time (s)')
            ax.set(ylabel='ECG(mv)')
            ax.plot(np.arange(0, 10, 1 / (128)), ecgSegs[idx[0]], "k", alpha=0.5)
            ax.scatter(np.arange(0, 10, 1 / (128))[r_peaks_list[idx[0]]], ecgSegs[idx[0]][r_peaks_list[idx[0]]])
            ax.legend(loc='upper left', prop={'size': 10})
            ax.set_xlim(left=0, right=10)
            ax.set_ylim(top=1.1, bottom=-0.1)
            # plt.title("{}:{:.3f}".format(ind, pred_curve[ind]))


            plt.show()


        hrv_deviation = (smoothed - hrv_threshold) / hrv_threshold
        for t in range(240):
            if hrv_deviation[t] > 0:
                # 指数修正
                adjustment = lambda_coef * hrv_deviation[t] + 0.2
                new_p1 = 0.5

                # 概率重分配
                delta_p = new_p1 - labels[t, 1]
                labels[t, 1] = new_p1
                labels[t, 0] -= delta_p


                # 概率裁剪
                labels[t] = np.clip(labels[t], 0, 1)

                # 归一化
                labels[t] /= labels[t].sum()
        plt.figure()
        plt.plot(labels[:, 0], "r")
        plt.plot(labels[:, 1], "k")
        plt.plot(labels[:, 2], "b")
        plt.show()
        return labels

def getHRVThreshold():
    trainPatient, valPatient, testPatient = GenerateTrainAndTest()
    trainPatient.extend(valPatient)
    S0 = []
    S01 = []
    S1 = []
    S2 = []
    for file_path in tqdm(trainPatient):
        with open(dataset_dir + file_path, 'rb') as file:
            file_data = pickle.load(file)
            data_ = file_data["X"]
            label_ = np.array(file_data["Y"]).reshape(-1, 1)
            if max(label_) > 0:
                S01.extend(data_[:60])
                pre_ids = np.where(label_ < 2)[0]
                pre_len = len(pre_ids)
                S1.extend(data_[pre_len-60:pre_len])
                S2.extend(data_[pre_len:pre_len + 60])
            else:
                S0.extend(data_[:60])
    data = []
    for i, ecglist in enumerate([S0, S01, S1, S2]):
        rmssd_list = []
        for ecg in ecglist:
            ecg = ecg.reshape(-1)
            rpeaks = getRpeak(ecg, 128)
            # 计算RR间期（单位：ms）
            # 计算时域特征
            rr_intervals = np.diff(rpeaks) / 128 * 1000  # 转换为毫秒
            try:

                # 计算RMSSD
                diffs = np.diff(rr_intervals)
                rmssd =  np.sqrt(np.mean(diffs ** 2))

            except:
                print("wrong")
                continue

            # 存储结果
            data.append({
                "Group": f"Group {i}",
                "RMSSD": rmssd
            })
            rmssd_list.append(rmssd)
        threshold_global = np.percentile(rmssd_list, 95)  # 取95%分位数
        print(threshold_global)
        threshold_global = np.mean(rmssd_list)  # 取95%分位数
        print(threshold_global)
    # 创建DataFrame
    df = pd.DataFrame(data)

    # 数据清洗（移除无效值）
    df_clean = df.dropna()

    # 绘制小提琴图
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="Group",
        y="RMSSD",
        data=df_clean,
        palette="Set3",
        inner="quartile"  # 显示四分位数线
    )

    plt.title("RMSSD Distribution Comparison", fontsize=14)
    plt.xlabel("Group", fontsize=12)
    plt.ylabel("RMSSD (ms)", fontsize=12)
    plt.xticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()







def status_dis_extraction():
    file_list = []
    trainPatient, valPatient, testPatient = GenerateTrainAndTest("inter")
    file_list.extend(trainPatient)
    file_list.extend(valPatient)
    file_list.extend(testPatient)

    # load model
    model_cls = indicator_cls_model(num_classes=3).to(DEVICE)
    pretrain_weight = torch.load(
        paf_model_save_path + f"classification_model_checkpoint/trs_status_dis_model_inter.pth",
        map_location=DEVICE)
    model_cls.load_state_dict(pretrain_weight, strict=True)
    model_cls.eval()

    model_status = MAE_linearmask(pre_train="train").to(DEVICE)
    model_status.load_state_dict(
        torch.load("/media/lzy/Elements SE/early_warning/pretrain_result/checkpoint/"
                   + f"mask_unmask_model_{100}.pth", map_location=DEVICE))
    model_status.eval()

    print(model_cls.class_center)

    indicator_data = {}
    status_data = {}
    prob_data = {}
    for file_path in tqdm(file_list):

        with open(dataset_dir + file_path, 'rb') as file:
            file_data = pickle.load(file)
            data_ = file_data["X"]
            label_ = np.array(file_data["Y"]).reshape(-1, 1)

            # if max(label_) > 1:
            #     probLabel = generateProbLabel(data_, label_)
            #
            # else:
            probLabel = [[1, 0, 0] for _ in label_]
            prob_data[file_path] = probLabel
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
            indicator_data[file_path] = [np.array(indicator_patient), label_, probLabel]

    data_save_path = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/"
    # np.save(data_save_path + "intra_prob.npy", prob_data)
    np.save(data_save_path + "inter_status_dis.npy", indicator_data)
    np.save(data_save_path + "inter_status.npy", status_data)


def spiltWindowData(sequence, label, status, win_size=224):
    start = 0
    data_len = len(sequence)
    stride = win_size // 2
    seq_seg = []
    label_seg = []
    status_seg = []

    assert len(sequence) == len(label) and len(sequence) == len(status)
    while start + win_size <= data_len:
        seq_seg.append(sequence[start: start + win_size])
        label_seg.append(label[start: start + win_size])
        status_seg.append(status[start: start + win_size])
        start += stride
    return np.array(seq_seg), np.array(label_seg), np.array(status_seg)


# data loader for warning indicator
class warningDataset(torch.utils.data.Dataset):
    def __init__(self, file_list=None, task="trainData", win_size=90, dataset_type=0):

        print("data loading start!")
        self.ind_all = []
        self.label_all = []
        self.status_all = []
        self.probLabel = []
        self.task = task
        self.dataset_type = dataset_type
        self.data_status = "train"

        status_data = np.load("/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/intra_status.npy",
                              allow_pickle=True).item()
        status_dis_data = np.load(
            "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/intra_status_dis.npy",
            allow_pickle=True).item()

        count = 0
        for file_path in tqdm(file_list):

            data_status = status_data[file_path].cpu().numpy()
            data_status_dis = status_dis_data[file_path][0]
            label_ori = status_dis_data[file_path][1].reshape(-1)
            probLabel = np.array(status_dis_data[file_path][2])

            if "PAF" in file_path:
                pre_ids = np.where(label_ori < 2)[0]
                pre_len = len(pre_ids)

                label_ori[pre_ids[-pre_len // 3:]] = 1

                # break
            status_patient = []
            status_dis_patient = []
            label_patient = []
            probLabel_patient = []
            for sta, sta_dis, la, prob in zip(data_status, data_status_dis, label_ori, probLabel):

                status_patient.append(sta)
                status_dis_patient.append(sta_dis)
                if "non" in file_path:
                    la = la + 1
                elif "PAF" in file_path:
                    la = la + 1
                else:
                    la = la
                label_patient.append(la)
                probLabel_patient.append(prob)



            self.status_all.append(np.array(status_patient))
            self.ind_all.append(np.array(status_dis_patient))
            self.probLabel.append(np.array(probLabel_patient))
            self.label_all.append(np.array(label_patient))

        print("丢弃的训练记录：{}/{}".format(count, len(file_list)))
        self.dataSet = []
        self.status = []
        self.label = []
        self.updateObservationLength(win_size)


    def generateLabel(self, label, start, end_ind):
        # 预警标签 低风险、中风险、高风险
        if max(label) == 0:
            return [1., 0., 0.]
        elif max(label) == 1:
            return [.7, 0.3, 0.]
        else:
            pre_status = np.where(label == 2)[0][0]
            if label[end_ind] == 1:
                ratio = (pre_status - end_ind) / pre_status
                if ratio > 0.75:
                    return [0.9, 0.1, 0.]
                else:
                    return [0.5, 0.5, 0.0]
            else:
                ratio = (end_ind - pre_status) / (end_ind - start)
                if ratio < 1:
                    return [0., 0.9, 0.1]
                elif ratio > 2:
                    return [0., 0.1, .9]
                else:
                    return [0., 0.5, 0.5]

    def spiltWindowData(self, sequence, label, status, probLabel=None, win_size=224):
        start = 0
        data_len = len(sequence)
        flag = False
        if max(label) < 2 or self.task == "testData":
            stride = 60
        else:
            stride = 12
            flag = True
        seq_seg = []
        label_seg = []
        status_seg = []

        assert len(sequence) == len(label) and len(sequence) == len(status)
        while start + win_size < data_len:
            seq_seg.append(sequence[start: start + win_size])
            label_seg.append(self.generateLabel(label, start, start + win_size))
            # label_seg.append(probLabel[start+win_size-1])
            status_seg.append(status[start: start + win_size])
            if flag:
                pre_status = np.where(label == 2)[0][0]
                if start + win_size >= pre_status and self.task == "trainData":
                    stride = 12
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
        for data_status, seq, lab, prob in tqdm(zip(self.status_all, self.ind_all, self.label_all, self.probLabel)):

            seq_seg, label_seg, status_seg = self.spiltWindowData(seq, lab, data_status, probLabel=prob, win_size=observationLength)
            seq_seg = seq_seg.reshape(-1, observationLength, 3)
            num_seg = len(seq_seg)
            # print(seq_seg.shape)

            ind[count: count + num_seg, :, :] = seq_seg
            label[count: count + num_seg, :] = label_seg
            status[count: count + num_seg, :, :, :] = status_seg
            count += num_seg
            assert count < initial_len
        print("什么情况")
        print(count)

        if self.dataset_type == 1:
            ind = ind[:count]
            status = status[:count]
            label = label[:count]
            print(np.sum(label, axis=0))
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
        d = torch.from_numpy(np.array(self.dataSet[item]))
        s = torch.from_numpy(np.array(self.status[item]))
        l = torch.from_numpy(np.array(self.label[item]))
        # if self.data_status == "val":
        #
        #     d = torch.from_numpy(np.array(self.dataSet[self.indices_test[item]]))
        #     s = torch.from_numpy(np.array(self.status[self.indices_test[item]]))
        #     l = torch.from_numpy(np.array(self.label[self.indices_test[item]]))
        # else:
        #     d = torch.from_numpy(np.array(self.dataSet[self.indices_train[item]]))
        #     s = torch.from_numpy(np.array(self.status[self.indices_train[item]]))
        #     l = torch.from_numpy(np.array(self.label[self.indices_train[item]]))
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
        self.lstm1 = nn.LSTM(input_size=fusion_dim, hidden_size=self.hidden_dim, batch_first=True, bidirectional=False,
                             num_layers=self.num_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, 3),

        )

    def soft_ce_loss(self, pred, tar, alpha=0.1):
        prob = F.softmax(pred, dim=1)
        prob_loss = alpha * (prob ** 2).sum(dim=1).mean()
        # 计算对数概率
        log_probs = F.log_softmax(pred, dim=-1)
        # 计算软标签的交叉熵损失
        # 这里用到的是对数概率和软标签直接的点积求和的负值
        loss = -(tar * log_probs).sum(dim=-1).mean()

        return loss + prob_loss

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
        # x = status
        x = self.ind_model(x[:, :, :])
        x = torch.cat([x, status], dim=2)

        lstm_output, hidden = self.lstm1(x, (h0, c0))
        latentFeature = self.avgpool(lstm_output.permute(0, 2, 1))
        # latentFeature = lstm_output[:, -1, :]
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
        # prob = nn.Softmax()(out_put)
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
            if threshold_value is not None:
                # score = np.mean(np.maximum(1 - abs(np.array(pred_time)[AF_idx] // 12 - 10) / 10, 0))
                score = np.mean(getTpScore(np.array(pred_time)[AF_idx] // 12))
                f_2 = (acc_list[0] * score * 5) / (acc_list[0] + 4 * score)
                print("测试H因子：{}".format(f_2))
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
        ratio = np.array(y1) / 12 / 30
        f_2 = (np.array(y2) * ratio * 5) / (np.array(y2) + 4 * ratio)
        ax.plot(x, f_2, label="F", color="blue", alpha=0.9, lw=1.0)
        ax.plot(x[[np.argmax(f_2), np.argmax(f_2)]], [-0.2, max(f_2)], "r--")
        print(x[np.argmax(f_2)])
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


def testClassification(model, dataloader):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for data_source in dataloader:
            ecg, = data_source[0].type(torch.FloatTensor).to(DEVICE),
            label = data_source[1]
            status = data_source[2].type(torch.FloatTensor).to(DEVICE)
            out_put, _ = model(ecg, status)
            _, predicted = torch.max(out_put.data, 1)
            true_labels += label.tolist()
            pred_labels += predicted.tolist()
    y_true_combined = np.array(true_labels).reshape(-1)
    y_pred_combined = np.array(pred_labels).reshape(-1)
    y_true_combined = np.where(y_true_combined == 2, 1, y_true_combined)
    # y_true_combined = np.where(y_true_combined == 3, 2, y_true_combined)
    y_pred_combined = np.where(y_pred_combined == 2, 1, y_pred_combined)
    # y_pred_combined = np.where(y_pred_combined == 3, 2, y_pred_combined)
    f, ax = plt.subplots()
    cm = confusion_matrix(y_true=y_true_combined, y_pred=y_pred_combined)
    sns.heatmap(cm, annot=True, ax=ax)  # 画热力图

    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x 轴
    ax.set_ylabel('true')  # y 轴
    # 命令行输出 混淆矩阵
    print('\nEvaluating....')
    print("TEST ACC:", accuracy_score(y_true_combined, y_pred_combined))
    print(classification_report(y_true_combined, y_pred_combined))
    print("Confusion Matrix:")
    print(cm)
    plt.show()


def getWarningRisk(model, status_dis, status, label, file_name=None):
    end_id = len(label)
    # end_ids = np.where(label==3)[0]
    # if len(end_ids) == 0:
    #     end_id = len(label) - 1
    # else:
    #     end_id = end_ids[0]

    risk_value = []
    label_true = []
    status_feature = torch.from_numpy(status[:end_id]).unsqueeze(0).to(DEVICE)
    status_dis = torch.from_numpy(status_dis[:end_id]).unsqueeze(0).to(DEVICE)
    initial_len = 11
    label = label[:end_id]

    data_pred = []
    data_label = []

    with torch.no_grad():
        pred_0, hidden = model(status_dis[:, :initial_len, :], status_feature[:, :initial_len, :, :])

        pred = nn.Softmax()(pred_0).cpu().numpy().reshape(3)
        data_pred.append(pred)
        data_label.append(label[initial_len - 1])
        r_v = 0.3 * pred[1] + 0.7 * pred[2]

        risk_value.append(r_v)
        label_true.append(label[initial_len - 1])
        for i in range(initial_len, len(label)):
            pred, hidden = model(status_dis[:, i, :].unsqueeze(1), status_feature[:, i, :, :].unsqueeze(1),
                                 hidden=hidden)
            pred = nn.Softmax()(pred).cpu().numpy().reshape(3)
            data_pred.append(pred)
            data_label.append(label[i])
            r_v = 0.3 * pred[1] + 0.7 * pred[2]

            risk_value.append(r_v)
            label_true.append(label[i])

    warning_curve_save_path = f"/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/warning_risk_value/ablation/{EXP_NAME}/"
    if not os.path.exists(warning_curve_save_path):
        os.makedirs(warning_curve_save_path)

    if file_name is not None:
        np.save(warning_curve_save_path + "{}.npy".format(file_name),
                {"pred": data_pred, "label": data_label})
    return risk_value, label_true


def saveWarningCurve(model):
    model.eval()
    trainData, valData, testData = GenerateTrainAndTest(EXP_NAME)
    trainData.extend(valData)

    print(" test patient", end=" ")
    print(testData)
    # data_save_path = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/indicator_preprocessing.pkl"
    status_data = np.load(f"/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/{EXP_NAME}_status.npy",
                          allow_pickle=True).item()
    status_dis_data = np.load(
        f"/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/{EXP_NAME}_status_dis.npy",
        allow_pickle=True).item()
    for j, train_patient in enumerate(trainData):

        data_status = status_data[train_patient].cpu().numpy()
        data_status_dis = np.array(status_dis_data[train_patient][0]).reshape(-1, 3)

        label_ori = status_dis_data[train_patient][1]
        if "PAF" in train_patient:
            pre_ids = np.where(label_ori < 2)[0]
            pre_len = len(pre_ids)
            label_ori[pre_ids[-pre_len // 3:]] = 1

        status_patient = []
        status_dis_patient = []
        label_patient = []
        for sta, sta_dis, la in zip(data_status, data_status_dis, label_ori):
            # if q == 0:
            status_patient.append(sta)
            status_dis_patient.append(sta_dis)
            if "PAF" in train_patient:
                la = la + 1
            else:
                la = la
            label_patient.append(la)

        pred_curve, true_curve = getWarningRisk(model, np.array(status_dis_patient),
                                                np.array(status_patient), np.array(label_patient),
                                                file_name=train_patient)

    for j, test_patient in enumerate(testData):
        data_status = status_data[test_patient].cpu().numpy()
        data_status_dis = np.array(status_dis_data[test_patient][0]).reshape(-1, 3)
        quality = status_dis_data[test_patient][2]
        label_ori = status_dis_data[test_patient][1]
        if "PAF" in test_patient:
            pre_ids = np.where(label_ori < 2)[0]
            pre_len = len(pre_ids)
            label_ori[pre_ids[-pre_len // 3:]] = 1

        status_patient = []
        status_dis_patient = []
        label_patient = []
        for sta, sta_dis, la, q in zip(data_status, data_status_dis, label_ori, quality):
            # if q == 0:
            status_patient.append(sta)
            status_dis_patient.append(sta_dis)
            if "non" in test_patient:
                la = la
            elif "PAF" in test_patient:
                la = la + 1
            else:
                la = la
            label_patient.append(la)

        pred_curve, true_curve = getWarningRisk(model, np.array(status_dis_patient), np.array(status_patient),
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
    # getHRVThreshold()
    # train_status_dis_model()
    # 仅调用一次
    # status_dis_extraction()
    EXP_NAME = "inter"
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    paf_model_save_path = "/media/lzy/Elements SE/early_warning/PAF_model/"
    finetune_epoch = 100
    flag = 2  # 0 warning training 1 testing 2 analysis

    if flag == 0:
        trainData, valData, testData = GenerateTrainAndTest("inter")
        trainData.extend(valData)
        dataloader = DataLoader(warningDataset(file_list=trainData, dataset_type=0), batch_size=16, shuffle=True)
        dataloader_test = DataLoader(warningDataset(file_list=valData, dataset_type=0), batch_size=64, shuffle=False)

        model = warningLSTM(fusion_dim=40).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epoch, eta_min=1e-4)

        val_loss = []
        train_loss = []
        val_loss_current = 1000000

        for epoch in range(1, finetune_epoch + 1):

            epoch_loss = training_one_epoch_model(model=model, dataloader=dataloader,
                                                  optimizer=optimizer, epoch=epoch,
                                                  max_epoch=finetune_epoch)
            scheduler.step()

            # dataloader.dataset.set_data_status("val")
            total_loss = evaluate_model(model, dataloader_test)
            # dataloader.dataset.set_data_status("train")
            train_loss.append(epoch_loss)
            val_loss.append(total_loss)
            if total_loss < val_loss_current:
                val_loss_current = total_loss
                save_path = paf_model_save_path + "warning_model_checkpoint/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                print('best_performance: {:.4f}'.format(total_loss))
                torch.save(model.state_dict(),
                           save_path + "warning_model_dis_inter.pth".format(epoch))

        fig, ax = plt.subplots(figsize=(4, 2), dpi=200)
        # ax.set_xticks(np.arange(0, 40, 4))
        # ax.set_yticks(np.arange(-0., 1.5, 0.4))
        ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
        ax.set(xlabel='time(min)')
        ax.set(ylabel='warning value')

        ax.plot(train_loss, label="train_loss", color="b", alpha=0.8, lw=0.6)
        ax.plot(val_loss, color="r",
                alpha=0.8, lw=0.5)

        ax.legend(loc='upper right', prop={'size': 7})

        #             plt.savefig("./图/PAFwarning_casestudy-N-n410.svg", format='svg')
        plt.show()

    elif flag == 1:
        model = warningLSTM(fusion_dim=40).to(DEVICE)
        pretrain_weight = torch.load(
            paf_model_save_path + f"warning_model_checkpoint/warning_model_dis_{EXP_NAME}.pth",
            map_location=DEVICE)
        model.load_state_dict(pretrain_weight, strict=True)
        model.eval()

        saveWarningCurve(model)

    # elif flag == 2:
        from AF_task.warningResultProcessing import TestThreshold, movingAvgWindow, getThreshold


        def warning_casestudy(PAF_patient, weight=0.3, threshold=0.33):
            # load ecg
            dataset_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/"
            with open(dataset_dir + PAF_patient, 'rb') as file:
                file_data = pickle.load(file)
                data_ = file_data["X"]
                # label_ = np.array(file_data["Y"]).reshape(-1, 1)
            print(data_.shape)
            warning_value = np.load(
                warning_curve_save_path +
                "/{}.npy".format(PAF_patient), allow_pickle=True).item()
            pred_value = np.array(warning_value["pred"])
            pred_curve = weight * pred_value[:, 1] + (1 - weight) * pred_value[:, 2]
            pred_curve = movingAvgWindow(pred_curve)
            print(pred_curve.shape)
            true_curve = np.array(warning_value["label"])
            ind = np.where(true_curve == 3)[0]
            if len(ind) == 0:
                onset_time = len(true_curve) - 1
            else:
                onset_time = ind[0]
            warning_ind = 0
            for i in range(60, len(pred_curve)):
                if pred_curve[i] >= threshold:
                    warning_ind = i
                    break
            print("pred_time: {}".format(warning_ind))

            ecg_length = len(data_) / 12
            ecg = list(data_[0])
            for j in range(1, len(data_)):
                ecg.extend(data_[j][640:])

            # plt.rcParams["font.family"] = "Calibri"
            fig, ax = plt.subplots(figsize=(4, 2), dpi=200)
            ax.set_xticks(np.arange(0, 40, 5))
            ax.set_yticks(np.arange(-0., 0.8, 0.3))
            ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
            ax.set(xlabel='time(min)')
            ax.set(ylabel='warning value')
            ax.plot(np.arange(10 / 12, ecg_length, 1 / 12), pred_curve, label="F", color="b", alpha=0.8, lw=0.6)
            ax.plot(np.arange(0, ecg_length + 1 / 12, 1 / 128 / 60), np.array(ecg) + .6, color="b", alpha=0.8, lw=0.05)

            print("预测平均值: {}".format(np.mean(pred_curve)))
            if "NSR" not in PAF_patient:
                ax.plot(np.arange(10 / 12, ecg_length, 1 / 12)[warning_ind:], pred_curve[warning_ind:], color="r",
                        alpha=0.8,
                        lw=0.4)
                ax.plot(np.arange(30, ecg_length + 1 / 12, 1 / 128 / 60), np.array(ecg)[30 * 128 * 60:] + .6, color="r",
                        alpha=0.8, lw=0.05)
                ax.plot([10 / 12 + onset_time / 12, 10 / 12 + onset_time / 12],
                        [-0.1, pred_curve[onset_time]], "r--", lw=0.3)

            ax.legend(loc='upper right', prop={'size': 7})
            ax.get_legend().remove()
            ax.set_xlim(left=0.1, right=ecg_length + 1)
            ax.set_ylim(top=1.65, bottom=-0.05)

            # plt.savefig(fig_save_path + f"warning_casestudy_{PAF_patient.split('.')[0]}.svg", format='svg')
            plt.show()

            if "NSR" in PAF_patient:
                dataset_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/"
                with open(dataset_dir + PAF_patient, 'rb') as file:
                    file_data = pickle.load(file)
                    data_ = file_data["X"]
                ecg = data_[120]
                with plt.style.context(['science', 'no-latex']):
                    fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
                    ax.minorticks_on()
                    ax.set_xticks(np.arange(0, 25, 1))
                    ax.set(xlabel='time (s)')
                    ax.set(ylabel='ECG(mv)')
                    ax.plot(np.arange(0, 10, 1 / (128)), ecg, "k", alpha=0.5)
                    ax.legend(loc='upper left', prop={'size': 10})
                    ax.set_xlim(left=0, right=10)
                    ax.set_ylim(top=1.1, bottom=-0.1)
                    # plt.title("{}:{:.3f}".format(ind, pred_curve[ind]))
                    # plt.savefig(fig_save_path + f"NSR_{PAF_patient}_ecg_{120}.svg", format='svg')

                    plt.show()


        fig_save_path = "/media/lzy/Elements SE/early_warning/revised_figure/"
        warning_curve_save_path = f"/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/warning_risk_value/ablation/{EXP_NAME}/"
        trainData, valData, testData = GenerateTrainAndTest(EXP_NAME)
        trainData.extend(valData)
        paf_dataset = []

        nsr_dataset = []
        print(" test patient", end=" ")
        print(testData)

        for file_name in trainData:

            if "PAF" in file_name:
                paf_dataset.append(file_name)

            else:
                nsr_dataset.append(file_name)

        print(len(paf_dataset))

        print(len(nsr_dataset))

        getThreshold(valData, path=warning_curve_save_path)
        TestThreshold(testData, 0.51, 0.3, path=warning_curve_save_path)
        for testP in testData[:5]:
            warning_casestudy(testP, weight=0.3,threshold=0.51)


    elif flag == 2:
        fig_save_path = "/media/lzy/Elements SE/early_warning/revised_figure/"
        def violin(all_data):
            fig, axs = plt.subplots(figsize=(4, 3), dpi=200)
            axs.grid(linestyle="--", linewidth=0.3)  # 设置背景网格线为虚线
            ax = plt.gca()
            ax.spines['top'].set_visible(False)  # 去掉上边框
            ax.spines['right'].set_visible(False)  # 去掉右边框

            enmax_palette = ["#F4C3DF",  "#B9C5FF"]

            sns.violinplot(data=all_data, cut=1.5,
                           scale='width', inner="box", linewidth=.6,
                           saturation=0.9, palette=enmax_palette, fill=False,
                           kde_kws={'bw_method': 0.1})
            # sns.boxplot(data=all_data,
            #             palette=enmax_palette,
            #             width=0.5,
            #             linewidth=.8,
            #             fliersize=2,  # 异常点大小
            #             whis=1.5,  # 须的长度倍数
            #             showfliers=True)
            # 计算并获取中位数
            for ii in range(2):
                median = np.median(all_data[ii])
                print(median)
                # 在图中标注中位数
                plt.text(ii - 0.1, median, '%.2f' % median, ha='center', va='bottom', size=6)
            # ax.plot([-2, 2.5], [71.06, 71.06], "r--", lw=1)

            group_labels = ['intra-patient', 'inter-patient']  # x轴刻度的标识
            plt.xticks(list([0, 1,]), group_labels, fontsize=6, fontweight='bold')  # 默认字体大小为10
            plt.yticks(fontsize=6, fontweight='bold')

            plt.ylabel("foretime", fontsize=6, fontweight='bold')
            plt.xlim(-0.8, 1.8)  # 设置x轴的范围
            # plt.ylim(0, 30)
            plt.savefig(fig_save_path + f'intra_inter_pred_time.svg', format='svg')
            plt.show()


        def warning_case_analysis(PAF_patient, weight=0.3):
            # load ecg
            dataset_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/"
            with open(dataset_dir + PAF_patient, 'rb') as file:
                file_data = pickle.load(file)
                data_ = file_data["X"]
            print(data_.shape)
            threshold_ls = [0.33, 0.23]
            pred_curve_list = []
            warning_ind_list = []
            onset_time_list = []
            for j, exp_ in enumerate(["intra", "inter"]):
                warning_path = f"/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/warning_risk_value/ablation/{exp_}/"
                warning_value = np.load(warning_path +"/{}.npy".format(PAF_patient), allow_pickle=True).item()
                pred_value = np.array(warning_value["pred"])
                pred_curve = weight * pred_value[:, 1] + (1 - weight) * pred_value[:, 2]
                pred_curve = movingAvgWindow(pred_curve)
                true_curve = np.array(warning_value["label"])
                ind = np.where(true_curve == 3)[0]
                if len(ind) == 0:
                    onset_time = len(true_curve) - 1
                else:
                    onset_time = ind[0]
                warning_ind = 0
                for i in range(60, len(pred_curve)):
                    if pred_curve[i] >= threshold_ls[j]:
                        warning_ind = i
                        break
                pred_curve_list.append(pred_curve)
                warning_ind_list.append(warning_ind)
                onset_time_list.append(onset_time)

            ecg_length = len(data_) / 12
            with plt.style.context(['science', 'no-latex']):
                fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
                ax.tick_params(top=False)
                ax.set(xlabel='time (min)')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                ax.plot(np.arange(10 / 12, ecg_length, 1 / 12), pred_curve_list[0], label="intra warning curve", color="k", alpha=0.8,
                        lw=0.9)
                ax.plot([10 / 12 + warning_ind_list[0] / 12, 10 / 12 + warning_ind_list[0] / 12],
                        [-0.1, pred_curve_list[0][warning_ind_list[0]]], "r--", lw=0.3)

                ax.plot(np.arange(10 / 12, ecg_length, 1 / 12), pred_curve_list[1], label="inter warning curve", color="b", alpha=0.8,
                        lw=0.9)

                ax.plot([10 / 12 + warning_ind_list[1] / 12, 10 / 12 + warning_ind_list[1] / 12],
                        [-0.1, pred_curve_list[1][warning_ind_list[1]]], "g--", lw=0.3)

                ax.set_ylim(top=0.55, bottom=0.1)
                ax.legend(loc='upper left', prop={'size': 10})
                # plt.savefig(fig_save_path + f"warning_curve_analysis_{PAF_patient.split('.')[0]}.svg", format='svg')
                plt.show()


            # for ind in [0, 110, 275, 288, warning_ind, 350]:
            #     ecg = data_[ind + 10]
            #     print("{}:{}".format(ind, pred_curve[ind]))
            #     with plt.style.context(['science', 'no-latex']):
            #         fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
            #         ax.minorticks_on()
            #         ax.set_xticks(np.arange(0, 25, 1))
            #         ax.set(xlabel='time (s)')
            #         ax.set(ylabel='ECG(mv)')
            #         ax.plot(np.arange(0, 10, 1 / (128)), ecg, "k", alpha=0.5)
            #         ax.legend(loc='upper left', prop={'size': 10})
            #         ax.set_xlim(left=0, right=10)
            #         ax.set_ylim(top=1.1, bottom=-0.1)
            #         plt.title("{}:{:.3f}".format(ind, pred_curve[ind]))
            #         plt.savefig(fig_save_path + f"warning_curve_analysis_ecg_{ind + 10}.svg", format='svg')
            #         plt.show()


        from AF_task.warningResultProcessing import TestThreshold, movingAvgWindow, getThreshold

        threshold_ls = [0.29, 0.23]
        fore_data = []
        patient_map = {}
        for i, exp_group in  enumerate(["intra", "inter"]):
            warning_curve_save_path = f"/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/warning_risk_value/ablation/{exp_group}/"
            trainData, valData, testData = GenerateTrainAndTest(exp_group)
            trainData.extend(valData)

            getThreshold(valData, path=warning_curve_save_path)

            acc, fore_time, pred_time , patient_af = TestThreshold(testData, threshold_ls[i], 0.3, path=warning_curve_save_path)
            fore_data.append(np.array(pred_time) / 12)

            for i, patient in enumerate(patient_af):
                if patient in patient_map:
                    patient_map[patient].append(pred_time[i] / 12)
                else:
                    patient_map[patient] = [pred_time[i] / 12]
        print(patient_map)
        for patient in patient_map.keys():
            if len(patient_map[patient]) == 2:
               if  patient_map[patient][0] < 12 and 15 < patient_map[patient][1] < 25:
                   warning_case_analysis(patient, weight=0.3)
        violin(fore_data)
        # hist(fore_data)



