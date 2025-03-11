import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from AF_task.PAF_warning_model import GenerateTrainAndTest
import scienceplots

warning_curve_save_path = "./AF_data/warning_risk_value/"

def TestThreshold(testData, threshold_value, weight, path = None):
    pred_val = []
    true_val = []
    onset_time = []
    patient_af = []
    for j, test_patient in enumerate(testData):
        if path is None:
            warning_value = np.load(
                warning_curve_save_path +
                "/{}.npy".format(test_patient), allow_pickle=True).item()
        else:
            warning_value = np.load(
                path +
                "/{}.npy".format(test_patient), allow_pickle=True).item()
        pred_value = np.array(warning_value["pred"])
        pred_curve = weight * pred_value[:, 1] + (1 - weight) * pred_value[:, 2]
        pred_curve = movingAvgWindow(pred_curve)
        true_curve = np.array(warning_value["label"])
        pred_val.append(pred_curve)
        if "PAF" in test_patient:
            true_val.append(1)
            ind = np.where(true_curve == 3)[0]
            if len(ind) == 0:
                onset_time.append(len(true_curve) - 1)
            else:
                onset_time.append(ind[0])
        else:
            true_val.append(0)
            onset_time.append(-1)
        warning_time = np.where(pred_curve > threshold_value)[0]
    acc_list = []
    forahead_time = []
    label_pred = []
    pred_time = []
    AF_pred_time = []
    for pred_patient, onset_time_patient in zip(pred_val, onset_time):
        AF_flag = False
        for i in range(50, len(pred_patient)):
            if pred_patient[i] >= threshold_value:
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
            patient_af.append(testData[kk])

    AF_pred_time = np.array(pred_time)[AF_idx]
    forahead_time.append(np.mean(np.array(pred_time)[AF_idx]))
    if threshold_value is not None:
        # score = np.mean(np.maximum(1 - abs(np.array(pred_time)[AF_idx] // 12 - 10) / 10, 0))
        score = np.mean(getTpScore(np.array(pred_time)[AF_idx] // 12))
        f_2 = (acc_list[0] * score * 5) / (acc_list[0] + 4 * score)
        print("测试集HM因子：{}".format(f_2))

    # 创建图表和第一个y轴

    # f, ax = plt.subplots()
    #
    plt.figure(1)
    x_tick = ["normal", 'PAF', ]
    y_tick = ["normal", "PAF"]
    plt.subplots(figsize=(2, 2), dpi=200)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    plt.grid = False
    #     mcm1 = multilabel_confusion_matrix(target, pred)
    #     vf_cm = [[8,0],[0,10]]
    sns.heatmap(confusion_matrix(true_val, label_pred), cmap='Blues', fmt='g', annot=True, cbar=False,
                xticklabels=x_tick, yticklabels=y_tick)

    plt.show()

    print("测试集预测精度：{}， 提前时间：{}".format(acc_list[0],  forahead_time[0]))
    return acc_list[0], forahead_time[0], AF_pred_time,

def movingAvgWindow(index_list):
    accuList = []
    win_size = 10
    for i in range(1, 1 + len(index_list)):

        if i < win_size:
            current = np.mean(index_list[:i])
        else:
            current = np.mean(index_list[i - win_size: i])
        accuList.append(current)
    return np.array(accuList)


def getTpScore(Tp, mu=10, sigma=5):
    score_list = []
    for s in Tp:
        if s == 0:
            score_list.append(0)
        else:
            score_list.append(np.exp(-((s - mu) ** 2) / (2 * sigma ** 2)))
    return score_list


def getThreshold(valData, path=None, save=None):
    weight_list = [0.3, 0.5,0.7]
    result_map = {}
    for key, weight in enumerate(weight_list):
        pred_val = []
        true_val = []
        onset_time = []
        acc_list = []
        forahead_time = []
        for j, val_patient in enumerate(valData):
            if path is None:
                warning_value = np.load(
                    warning_curve_save_path +
                    "/{}.npy".format(val_patient), allow_pickle=True).item()
            else:
                warning_value = np.load(
                    path +
                    "/{}.npy".format(val_patient), allow_pickle=True).item()
            pred_value = np.array(warning_value["pred"])
            pred_curve = weight * pred_value[:, 1] + (1 - weight) * pred_value[:, 2]
            pred_curve = movingAvgWindow(pred_curve)
            true_curve = np.array(warning_value["label"])
            pred_val.append(pred_curve)
            if "PAF" in val_patient:
                true_val.append(1)
                ind = np.where(true_curve == 3)[0]
                if len(ind) == 0:
                    onset_time.append(len(true_curve)-1)
                else:
                    onset_time.append(ind[0])
            else:
                true_val.append(0)
                onset_time.append(-1)
        for threshold in np.arange(0.05, .9, 0.02):
            label_pred = []

            pred_time = []
            for pred_patient, onset_time_patient in zip(pred_val, onset_time):
                AF_flag = False
                for i in range(60, len(pred_patient)):
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
                # score = np.mean(np.maximum(1 - abs(np.array(pred_time)[AF_idx] // 12 - 10) / 10, 0))
                score = np.mean(getTpScore(np.array(pred_time)[AF_idx] // 12))
                forahead_time.append(score)

        result_map[key] = [forahead_time, acc_list]

    x = np.arange(0.05, .9, 0.02)
    # plt.rcParams["font.family"] = "Calibri"
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
        ax.set_xticks(np.arange(0, 0.6, 0.25))
        ax.set_yticks(np.arange(-0.0, 1., 0.2))
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
        ax.set(xlabel='threshold')
        ax.set(ylabel='HM value')
        for key in range(len(weight_list)):
            [forahead_time, acc_list] = result_map[key]
            hm = (np.array(acc_list) * np.array(forahead_time) * 5) / (np.array(acc_list) + 4 * np.array(forahead_time))

            ax.plot(x, hm, label=f"weight-{weight_list[key]}", alpha=0.9, lw=1.)
            ax.plot(x[[np.argmax(hm), np.argmax(hm)]], [-0.2, max(hm)], "r--")
            print(f"weight-{weight_list[key]}:{x[np.argmax(hm)]}, {max(hm)} ")
    ax.set_xlim(left=0.1, right=.6)
    ax.set_ylim(top=1., bottom=-0.1)
    plt.legend()
    plt.show()

def warning_test_result(testData):
    ratio_list = []
    for (threshold, weight) in [(0.33, 0.3), (0.49, 0.5), (0.61, 0.7)]:
        acc_test, advance_time, AF_pred_time = TestThreshold(testData, threshold_value=threshold, weight=weight)
        ratios = []
        # for (s,e) in [(0,5), (5,10), (10,15), (15,20),(20,25), (25,30)]:
        #     ratios.append(sum(1 for x in AF_pred_time if (s <= x / 12 < e)) / 78)
        for s in range(30):
            ratios.append(sum(1 for x in AF_pred_time if (s <= x / 12 )) / 38)
        ratio_list.append(ratios)

    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
        ax.set_xticks(np.arange(0, 30, 1))
        ax.set_yticks(np.arange(-0.0, 1., 0.2))
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
        ax.set(xlabel='time before AF onset (min)')
        ax.set(ylabel='The proportion of predicted AF')
        for i, weight in enumerate([0.3, 0.5, 0.7]):
            # 柱状图
            ax.plot( range(30), ratio_list[i], alpha=0.6, label=f'weight-{weight}')
        plt.xticks([0, 5, 10, 15, 20, 25, 30])
        plt.legend()

        plt.show()




if __name__ =="__main__":

    trainData, valData, testData = GenerateTrainAndTest()
    trainData.extend(valData)
    trainData.extend(testData)
    # dataloader = DataLoader(warningDataset(file_list=testData, task="testData"), batch_size=32, shuffle=False)
    # testClassification(model, dataloader)
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

    # get threshold
    getThreshold(trainData)
    # warning_test_result(valData)
    TestThreshold(testData, 0.33, 0.3)

