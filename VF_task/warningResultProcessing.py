import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

import scienceplots
fig_save_path = "/media/lzy/Elements SE/early_warning/revised_figure/VF/"
warning_curve_save_path = "/media/lzy/Elements SE/early_warning/VF_data/warning_risk_value/"
dataset_dir = "/media/lzy/Elements SE/early_warning/VF_data/"

def TestThreshold(testData, threshold_value, weight):

    pred_val = []
    true_val = []
    onset_time = []

    for j, test_patient in enumerate(testData):

        warning_value = np.load(
            warning_curve_save_path +
            "/{}.npy".format(test_patient), allow_pickle=True).item()
        pred_value = np.array(warning_value["pred"])
        pred_curve = weight * pred_value[:, 1] + (1 - weight) * pred_value[:, 2]
        pred_curve = movingAvgWindow(pred_curve)
        true_curve = np.array(warning_value["label"])
        pred_val.append(pred_curve)
        if "VF" in test_patient:
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

        # plt.title(test_patient)
        # plt.plot(range(len(true_curve)), np.array(true_curve) / 3, label="prob", c="red", marker="x")
        # plt.plot(range(len(true_curve)), pred_curve, label="pred", )
        # plt.plot(range(len(true_curve)), [threshold_value] * len(true_curve), 'r--', )
        # if len(warning_time) > 0:
        #     warning_time = warning_time[0]
        #     plt.plot([warning_time, warning_time], [0, threshold_value], 'k--', )
        # plt.ylim(-0.2, 1.5)
        # plt.legend()
        # plt.savefig(dataset_dir +
        #             "warning_result/{}.png".format(test_patient), format='png')
        # plt.show()

    acc_list = []
    forahead_time = []
    label_pred = []
    pred_time = []
    AF_pred_time = []
    for pred_patient, onset_time_patient in zip(pred_val, onset_time):
        AF_flag = False
        for i in range(len(pred_patient)):
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
    x_tick = ["normal", 'VF', ]
    y_tick = ["normal", "VF"]
    plt.subplots(figsize=(2, 2), dpi=200)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    plt.grid = False
    #     mcm1 = multilabel_confusion_matrix(target, pred)
    #     vf_cm = [[8,0],[0,10]]
    sns.heatmap(confusion_matrix(true_val, label_pred), cmap='Blues', fmt='g', annot=True, cbar=False,
                xticklabels=x_tick, yticklabels=y_tick)
    # plt.savefig(fig_save_path + "vf_warning_cm.svg", format='svg')
    plt.show()
    # 命令行输出 混淆矩阵
    # print('\nEvaluating....')
    # print("TEST ACC:", accuracy_score(true_val, label_pred))
    # print(classification_report(true_val, label_pred))
    # print("Confusion Matrix:")
    # print(cm)
    # plt.show()
    print("测试集预测精度：{}， 提前时间：{}".format(acc_list[0],  forahead_time[0]))
    return acc_list[0], forahead_time[0], AF_pred_time

def movingAvgWindow(index_list):
    accuList = []
    win_size = 5
    for i in range(1, 1 + len(index_list)):

        if i < win_size:
            current = np.sum(index_list[:i]) / win_size
        else:
            current = np.mean(index_list[i - win_size: i])
        accuList.append(current)
    return np.array(accuList)


def getTpScore(Tp, mu=5, sigma=2):
    score_list = []
    for s in Tp:
        if s == 0:
            score_list.append(0)
        else:
            score_list.append(np.exp(-((s - mu) ** 2) / (2 * sigma ** 2)))
    return score_list


def getThreshold(valData):
    weight_list = [0.3, 0.5]
    result_map = {}
    for key, weight in enumerate(weight_list):
        pred_val = []
        true_val = []
        onset_time = []
        acc_list = []
        forahead_time = []
        for j, val_patient in enumerate(valData):

            warning_value = np.load(
                warning_curve_save_path +
                "/{}.npy".format(val_patient), allow_pickle=True).item()
            pred_value = np.array(warning_value["pred"])
            pred_curve = weight * pred_value[:, 1] + (1 - weight) * pred_value[:, 2]
            pred_curve = movingAvgWindow(pred_curve)
            true_curve = np.array(warning_value["label"])
            pred_val.append(pred_curve)
            if "VF" in val_patient:
                true_val.append(1)
                ind = np.where(true_curve == 3)[0]
                if len(ind) == 0:
                    onset_time.append(len(true_curve)-1)
                else:
                    onset_time.append(ind[0])
            else:
                true_val.append(0)
                onset_time.append(-1)
        for threshold in np.arange(0.05, .72, 0.02):
            label_pred = []

            pred_time = []
            for pred_patient, onset_time_patient in zip(pred_val, onset_time):
                VF_flag = False
                for i in range(len(pred_patient)):
                    if pred_patient[i] >= threshold:
                        pred_time.append(max(onset_time_patient - i, 0))
                        VF_flag = True
                        label_pred.append(1)
                        break
                if not VF_flag:
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
                # score = np.mean(np.maximum(1 - abs(np.array(pred_time)[AF_idx] // 12 - 10) / 10, 0))
                score = np.mean(getTpScore(np.array(pred_time)[VF_idx] // 12))
                forahead_time.append(score)

        result_map[key] = [forahead_time, acc_list]

    x = np.arange(0.05, .72, 0.02)
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
            # ax.plot(x[[np.argmax(hm), np.argmax(hm)]], [-0.2, max(hm)], "r--")
            print(f"weight-{weight_list[key]}:{x[np.argmax(hm)]}, {max(hm)} ")
    ax.set_xlim(left=0.1, right=.72)
    ax.set_ylim(top=1., bottom=-0.1)
    plt.legend()
    plt.savefig(fig_save_path + "HM_curve_weight_3.svg", format='svg')
    plt.show()

def warning_test_result(testData):
    ratio_list = []
    for (threshold, weight) in [(0.35, 0.3), (0.43, 0.5), (0.51, 0.7)]:
        acc_test, advance_time, AF_pred_time = TestThreshold(testData, threshold_value=threshold, weight=weight)
        ratios = []

        for s in range(15):
            ratios.append(sum(1 for x in AF_pred_time if (s <= x / 12)) / 17)
        ratio_list.append(ratios)

    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(figsize=(5, 3), dpi=200)
        ax.set_xticks(np.arange(0, 15, 1))
        ax.set_yticks(np.arange(-0.0, 1.1, 0.2))
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
        ax.set(xlabel='time before VF onset (min)')
        ax.set(ylabel='The proportion of predicted VF')
        for i, weight in enumerate([0.3, 0.5, 0.7]):
            # 柱状图
            ax.plot(range(15), ratio_list[i], alpha=0.6, label=f'weight-{weight}')
        plt.xticks([0, 5, 10, 15])
        plt.legend()
        plt.savefig(fig_save_path + "proportion_Ta.svg", format='svg')
        plt.show()


def warning_casestudy(VF_patient, weight=0.3, threshold=0.33):
    # load ecg
    dataset_dir = "/media/lzy/Elements SE/early_warning/VF_data"
    file_path = dataset_dir + "/VF_all_data.npy"

    file_data = np.load(file_path, allow_pickle=True).item()["testData"]
    data_ = file_data[VF_patient]["X"]

    warning_value = np.load(
        warning_curve_save_path +
        "/{}.npy".format(VF_patient), allow_pickle=True).item()
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
    for i in range(len(pred_curve)):
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
    ax.set_xticks(np.arange(0, 10, 2))
    ax.set_yticks(np.arange(-0., 0.8, 0.3))
    ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
    ax.set(xlabel='time(min)')
    ax.set(ylabel='warning value')
    ax.plot(np.arange(4/12, ecg_length, 1 / 12)[:len(pred_curve)], pred_curve, label="F", color="b", alpha=0.8, lw=0.6)
    ax.plot(np.arange(0, ecg_length + 1/12, 1 / 128 / 60), np.array(ecg) + .6, color="b", alpha=0.8, lw=0.05)

    print("预测平均值: {}".format(np.mean(pred_curve)))
    if "NSR" not in VF_patient:
        ax.plot(np.arange(4 / 12, ecg_length, 1 / 12)[warning_ind:], pred_curve[warning_ind:], color="r", alpha=0.8,
                lw=0.4)
        ax.plot(np.arange(4/12 + onset_time / 12, ecg_length + 1/12, 1 / 128 / 60),
                np.array(ecg)[int((4/12 + onset_time / 12) * 128 * 60):] + .6, color="r",
                alpha=0.8, lw=0.05)
        ax.plot([4/12 + onset_time / 12, 4/12 + onset_time / 12],
                [-0.1, pred_curve[onset_time]], "r--", lw=0.3)

    ax.legend(loc='upper right', prop={'size': 7})
    ax.get_legend().remove()
    ax.set_xlim(left=0.1, right=8+1)
    ax.set_ylim(top=1.65, bottom=-0.05)

    # plt.savefig(fig_save_path + f"warning_casestudy_{VF_patient}.svg", format='svg')
    plt.show()

    if "NSR" in VF_patient:
        ecg = data_[90]
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
            plt.savefig(fig_save_path + f"NSR_{VF_patient}_ecg_{90}.svg", format='svg')

            plt.show()


def warning_case_analysis(VF_patient, weight=0.3, threshold=0.33):
    # load ecg
    dataset_dir = "/media/lzy/Elements SE/early_warning/VF_data"
    file_path = dataset_dir + "/VF_all_data.npy"

    file_data = np.load(file_path, allow_pickle=True).item()["testData"]
    data_ = file_data[VF_patient]["X"]

    warning_value = np.load(
        warning_curve_save_path +
        "/{}.npy".format(VF_patient), allow_pickle=True).item()
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
    for i in range(len(pred_curve)):
        if pred_curve[i] >= threshold:
            warning_ind = i
            break
    print(warning_ind)
    ecg_length = len(data_) / 12
    with plt.style.context(['science', 'no-latex']):
        fig, ax = plt.subplots(figsize=(6, 2), dpi=200)
        #             ax.minorticks_on()
        #             ax.set_xticks(np.arange(0, 45, 5))
        # ax.set_yticks([])
        ax.tick_params(top=False)
        #             ax.grid(which='major', linestyle='--', linewidth='0.1', color='k')
        ax.set(xlabel='time (min)')

        #             ax.set(ylabel='warning value')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)

        ax.plot(np.arange(4/12, ecg_length, 1 / 12)[:len(pred_curve)], pred_curve, label="F", color="b", alpha=0.8, lw=0.6)
        ax.plot(np.arange(4 / 12, ecg_length, 1 / 12)[warning_ind:], pred_curve[warning_ind:], color="r", alpha=0.8,
                lw=0.4)
        ax.plot([4 / 12 + onset_time / 12, 4 / 12 + onset_time / 12],
                [-0.1, pred_curve[onset_time]], "r--", lw=0.3)
        ax.set_ylim(top=0.75, bottom=0.1)
        ax.legend(loc='upper left', prop={'size': 10})
        # plt.savefig(fig_save_path + f"warning_curve_analysis_{VF_patient}.svg", format='svg')
        plt.show()


    # ecg_length = len(data_) / 12
    # ecg = list(data_[0])
    # for j in range(1, len(data_)):
    #     ecg.extend(data_[j][640:])
    for ind in [0, 12, 36, warning_ind, 72]:
        ecg = data_[ind + 4]

        print("{}:{}".format(ind, pred_curve[ind]))
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
            plt.title("{}:{:.3f}".format(ind, pred_curve[ind]))
            # plt.savefig(fig_save_path + f"warning_curve_analysis_ecg_{ind+4}.svg", format='svg')

            plt.show()



if __name__ =="__main__":



    status_dis_data = np.load(dataset_dir + "status_dis.npy",
                              allow_pickle=True).item()

    trainData = status_dis_data["trainData"].keys()
    testData = status_dis_data["testData"].keys()
    # get threshold

    vf_dataset = []

    nsr_dataset = []
    print(" test patient", end=" ")
    print(testData)

    for file_name in trainData:

        if "VF" in file_name:
            vf_dataset.append(file_name)
        else:
            nsr_dataset.append(file_name)

    print(len(vf_dataset))

    print(len(nsr_dataset))

    # get threshold
    # getThreshold(trainData)
    # warning_test_result(trainData)
    # TestThreshold(testData, 0.61, 0.9)
    # #"PAF_record_001_0.pkl",
    for test_patient in [#"VF_cu13", #"VF_cu22",
                         "NSR_17453_nsr_6.pkl"]:
        warning_casestudy(test_patient)
    # warning_case_analysis("VF_cu13")
    # warning_case_analysis("NSR_n49.pkl")

