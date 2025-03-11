
import os

import pickle
from collections import Counter

import random
import wfdb
from sklearn.model_selection import train_test_split
import scipy
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from utils.dfilters import IIRRemoveBL


def get_paf2001_data():
    path = "/home/lzy/workspace/dataSet/paf-prediction-challenge-database-1.0.0/paf-prediction-challenge-database-1.0.0/"
    out_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/"
    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')
    dataSet = []
    for record_name in all_record_name:
        if ("p" in record_name or "n" in record_name) and "c" not in record_name:
            num = int(record_name[1:])
            if num % 2 == 1:
                dataSet.append(record_name)
    paf_data = {}
    non_paf_data = {}
    annotate_rs = np.load("/media/lzy/Elements SE/early_warning/PAF_data/paf_2001_lead_annotation.npy", allow_pickle=True).item()
    print(annotate_rs)
    for recordName in tqdm.tqdm(dataSet):
        if "n27" in recordName:
            continue
        num = int(recordName[1:])
        ecg4patient = []
        if "p" in recordName:
            # paf normal phase
            sig, fields = wfdb.rdsamp(path + recordName)
            lead = annotate_rs[recordName]
            fs = fields['fs']
            sig = sig[:, int(lead)]
            non_paf_data["non_" + recordName] = [sig]

            # paf pre phase
            sig, fields = wfdb.rdsamp(path + "p{:02d}".format(num+1))
            lead = annotate_rs["p{:02d}".format(num+1)]
            sig = sig[:, int(lead)]
            ecg4patient.append(sig)


            paf_data[recordName] = ecg4patient
        else:
            sig, fields = wfdb.rdsamp(path + recordName)
            lead = annotate_rs[recordName]

            fs = fields['fs']
            sig = sig[:, int(lead)]
            ecg4patient.append(sig)
            non_paf_data[recordName] = ecg4patient



    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data4save = {"PAF": paf_data, "NSR": non_paf_data, "fs": fs}
    with open(out_dir + "orginal_paf_2001.pkl", 'wb') as file:
        pickle.dump(data4save, file)


def normalize_linear(arr):
    # 计算数组每一行的最小值和最大值
    row_min = np.min(arr, keepdims=True)
    row_max = np.max(arr, keepdims=True)
    # 将数组每一行进行线性归一化
    normalize_flag = (row_max == row_min)
    row_max[normalize_flag] = 1
    row_min[normalize_flag] = 0
    arr_normalized = (arr - row_min) / (row_max - row_min)
    # 将最大最小值相等的行还原为原始值
    arr_normalized[normalize_flag.squeeze()] = arr[normalize_flag.squeeze()]

    return arr_normalized


def getNSRData():
    path = '/home/lzy/workspace/codeFile/VFearlywarning/mit-bih-normal-sinus-rhythm-database-1.0.0'
    out_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/"
    save_path = out_dir + "paf_warning_dataset/"

    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')

    for record_name in all_record_name:
        tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
        fs = tmp_data_res[1]['fs']
        window_size = int(fs * 10)
        stride = int(fs * 5)
        segment_time_stamp = random.sample(list(range(24)), 5)

        for time_stamp in segment_time_stamp:
            pp_data = []
            pp_label = []
            channel = 0
            idx_start = 0
            tmp_data = tmp_data_res[0][:, channel][(time_stamp * 60) * fs: (time_stamp * 60 + 30 * 60) * fs]

            tmp_data = IIRRemoveBL(tmp_data, fs, Fc=0.67)
            while idx_start <= len(tmp_data) - window_size:
                idx_end = idx_start + window_size
                if fs != 128:
                    ecg_seg = scipy.signal.resample(tmp_data[idx_start:idx_end], 128 * 10)
                else:
                    ecg_seg = tmp_data[idx_start:idx_end]
                ecg_seg = normalize_linear(ecg_seg)
                pp_data.append(ecg_seg)
                pp_label.append(0)
                idx_start += stride
            with open(save_path + "NSR_{}_nsr_{}.pkl".format(record_name, time_stamp), 'wb') as file:
                print(len(pp_data))
                pickle.dump({"X": np.array(pp_data), "Y": np.array(pp_label)}, file)

def data_preprocessing():
    out_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/"
    save_path = out_dir + "paf_warning_dataset/"

    with open(out_dir + "orginal_paf_2001.pkl", 'rb') as file:
        data_paf_2001 = pickle.load(file)
    win_size = data_paf_2001["fs"] * 10
    stride = data_paf_2001["fs"] * 5
    for key in ["PAF", "NSR"]:
        for patient_name in tqdm.tqdm(data_paf_2001[key].keys()):
            data_patient = []
            data_label = []
            data_quality = []
            count_bad_quality = 0
            for i, ecg_phase in enumerate(data_paf_2001[key][patient_name]):
                ecg_phase = IIRRemoveBL(ecg_phase, data_paf_2001["fs"], Fc=0.67)
                start = 0
                data_len = len(ecg_phase)
                if key == "NSR" and i > 0:
                    continue
                while start + win_size < data_len:

                    ecg_seg = ecg_phase[start: start + win_size]
                    ecg_seg = normalize_linear(ecg_seg)
                    if data_paf_2001["fs"] == 128:
                        pass
                    else:
                        ecg_seg = scipy.signal.resample(ecg_seg, 128 * 10)
                    idx_end = start + win_size
                    if key == "PAF":
                        if i == 0:
                            if  idx_end <= 20 * 60 * data_paf_2001["fs"]:
                                label = 0
                            else:
                                label = 1
                        else:
                            label = 2
                    else:
                        label = 0
                    data_patient.append(ecg_seg)
                    data_label.append(label)
                    start += stride

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(save_path + "{}_{}.pkl".format(key, patient_name), 'wb') as file:
                print(len(data_patient))
                pickle.dump({"X": np.array(data_patient), "Y": np.array(data_label), "Q": np.array(data_quality)}, file)


    #
    iridia_dir = "/media/lzy/Elements SE/early_warning/PAF_data/iridia/"
    iridia_files = os.listdir(iridia_dir)
    # 使用列表推导式筛选出以'.mp3'结尾的文件名
    iridia_files = [filename for filename in iridia_files if filename.endswith('.pkl')]
    win_size = 200 * 10
    stride = 200 * 5
    Fs = 200
    for patient_name in tqdm.tqdm(iridia_files):
        data_patient = []
        data_label = []
        with open(iridia_dir + patient_name, 'rb') as file:
            data_paf_iridia = pickle.load(file)["data"]
        for i, ecg_phase in enumerate(data_paf_iridia):

            ecg_phase = IIRRemoveBL(ecg_phase, Fs, Fc=0.67)

            start = 0
            data_len = len(ecg_phase)

            while start + win_size < data_len:

                ecg_seg = ecg_phase[start: start + win_size]
                ecg_seg = normalize_linear(ecg_seg)
                if Fs == 128:
                    pass
                else:
                    ecg_seg = scipy.signal.resample(ecg_seg, 128 * 10)

                idx_end = start + win_size
                if i == 0:
                    if idx_end <= 20 * 60 * Fs:
                        label = 0
                    else:
                        label = 1
                else:
                    label = 2

                data_patient.append(ecg_seg)
                data_label.append(label)
                start += stride
        with open(save_path + "PAF_{}".format(patient_name), 'wb') as file:
            pickle.dump({"X": np.array(data_patient), "Y": np.array(data_label)}, file)

def ECGplot(sigg, new_sig=None, title=None):
    fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
    #     ax.set_xticks(np.arange(0, 12, 0.5))
    ax.set_yticks(np.arange(-5.0, +5.0, 0.5))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
    t = np.arange(0, len(sigg) * 1 / 200, 1 / 200)

    ax.plot(t, sigg, label="ECG signal", color='k')
    ymin = -1.5
    ymax = 2.5
    if new_sig is not None:
        ax.plot(t, new_sig + 2, label="new ECG signal", color='g')
        ymax = ymax + 2
    ax.legend(loc='upper right', bbox_to_anchor=(.96, 1.0))
    ax.set(xlabel='time (s)')
    #     ax.set(ylabel='Voltage (mV)')
    ax.autoscale(tight=True)
    ax.set_xlim(left=0, right=10.5)
    plt.title(title)
    ax.set_ylim(top=ymax, bottom=ymin)
    # plt.savefig(i_path)
    plt.show()


def seed_torch(seed=1029):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)


def GenerateTrainAndTest():
    dataset_dir = "./AF_data/"
    dataset_files = os.listdir(dataset_dir)
    # 使用列表推导式筛选出以'.mp3'结尾的文件名
    dataset_files = [filename for filename in dataset_files if filename.endswith('.pkl') ]
    if "train_test_spilt.pkl" in dataset_files:
        with open(dataset_dir + "train_test_spilt.pkl", 'rb') as file:
            patient = pickle.load(file)
            print(len(patient["trainPatient"]))
            print(len(patient["valPatient"]))
            print(patient["testPatient"])

            return patient["trainPatient"], patient["valPatient"], patient["testPatient"]

    seed_torch(123)

    trainData, testData = train_test_split(dataset_files, test_size=0.3, random_state=111)
    trainData, valData = train_test_split(trainData, test_size=0.2, random_state=111)
    with open(dataset_dir + "train_test_spilt.pkl", 'wb') as file:
        pickle.dump({"trainPatient": trainData, "valPatient": valData, "testPatient": testData}, file)
    return trainData, valData, testData

def data_vis():

    dataset_dir = "./AF_data/"

    with open(dataset_dir + "PAF_record_158_4.pkl", 'rb') as file:
        file_data = pickle.load(file)
        data_paf_iridia = file_data["X"]
        label =  file_data["Y"]
    print(data_paf_iridia.shape)
    print(Counter(label))
    plt.figure()
    plt.plot(data_paf_iridia[-1], "r")
    plt.plot(data_paf_iridia[0], "b")
    plt.show()

if __name__ == "__main__":

    # getNSRData()
    # get_paf2001_data()
    # data_preprocessing()
    GenerateTrainAndTest()
    data_vis()




























