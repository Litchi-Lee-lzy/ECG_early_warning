
import os

import pickle
from collections import Counter
import random

import joblib
import wfdb
from sklearn.model_selection import train_test_split
import scipy
import numpy as np
import matplotlib.pyplot as plt

from utils.dfilters import IIRRemoveBL

save_path = "/media/lzy/Elements SE/early_warning/VF_data/"

def get_label_map(labels):
    m = {'N': 'SN',  # Normal beat (displayed as "·" by the PhysioBank ATM, LightWAVE, pschart, and psfd)
         'L': 'LBBB',  # Left bundle branch block beat
         'R': 'RBBB',  # Right bundle branch block beat
         'B': 'IVB',  # Bundle branch block beat (unspecified)
         'A': 'PAC',  # Atrial premature beat
         'a': 'PAC',  # Aberrated atrial premature beat
         'J': 'PJC',  # Nodal (junctional) premature beat
         'S': 'PSC',  # Supraventricular premature or ectopic beat (atrial or nodal)
         'V': 'PVC',  # Premature ventricular contraction
         'r': 'PVC',  # R-on-T premature ventricular contraction
         'F': 'PVC',  # Fusion of ventricular and normal beat
         'e': 'AE',  # Atrial escape beat
         'j': 'JE',  # Nodal (junctional) escape beat
         'n': 'SE',  # Supraventricular escape beat (atrial or nodal)
         'E': 'VE',  # Ventricular escape beat
         '/': 'PACED',  # Paced beat
         'f': 'PACED',  # Fusion of paced and normal beat
         'Q': 'OTHER',  # Unclassifiable beat
         '?': 'OTHER',  # Beat not classified during learning
         '[': 'VF',  # Start of ventricular flutter/fibrillation
         '!': 'VF',  # Ventricular flutter wave
         ']': 'VF',  # End of ventricular flutter/fibrillation
         'x': 'PAC',  # Non-conducted P-wave (blocked APC)
         '(AB': 'PAC',  # Atrial bigeminy
         '(AFIB': 'AF',  # Atrial fibrillation
         '(AF': 'AF',  # Atrial fibrillation
         '(AFL': 'AFL',  # Atrial flutter
         '(ASYS': 'PAUSE',  # asystole
         '(B': 'PVC',  # Ventricular bigeminy
         '(BI': 'AVBI',  # 1° heart block
         '(BII': 'AVBII',  # 2° heart block
         '(HGEA': 'PVC',  # high grade ventricular ectopic activity
         '(IVR': 'VE',  # Idioventricular rhythm
         '(N': 'SN',  # Normal sinus rhythm
         '(NOD': 'JE',  # Nodal (A-V junctional) rhythm
         '(P': 'PACED',  # Paced rhythm
         '(PM': 'PACED',  # Paced rhythm
         '(PREX': 'WPW',  # Pre-excitation (WPW)
         '(SBR': 'SNB',  # Sinus bradycardia
         '(SVTA': 'SVT',  # Supraventricular tachyarrhythmia
         '(T': 'PVC',  # Ventricular trigeminy
         '(VER': 'VE',  # ventricular escape rhythm
         '(VF': 'VF',  # Ventricular fibrillation
         '(VFL': 'VFL',  # Ventricular flutter
         '(VT': 'VT'  # Ventricular tachycardia
         }
    out_labels = []
    for i in labels:
        if i in m:
            if m[i] in ['VF', 'VFL',  '(VF', '(VT', "VT"]:
                out_labels.append('VF/VT')
            else:
                out_labels.append("Others")
    out_labels = list(np.unique(out_labels))
    return out_labels
def getCUDB():
    valid_lead = ['MLII', 'ECG', 'V5', 'V2', 'V1', 'ECG1', 'ECG2' ] # extract all similar leads
    t = 10
    window_size_t = 10 # second
    stride_t = 5 # second
    fs_out = 128
    all_pid = []
    all_data = {}
    VF_PRE_DURATION = []
    path = '/home/lzy/workspace/codeFile/VFearlywarning/cu-ventricular-tachyarrhythmia-database-1.0.0'
    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')
    for record_name in all_record_name:
        if record_name in ["cu12", "cu15", "cu24", "cu25", "cu32"]:
            print("起搏心律")
            continue
        # if record_name in ["cu02", "cu03", "cu06", "cu09", "cu12", "cu21", "cu27", "cu28", "cu35"]:
        #     print("室速心律")
        #     continue
        cnt = 0
        tmp_ann_res = wfdb.rdann(path + '/' + record_name, 'atr').__dict__
        tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
        fs = tmp_data_res[1]['fs']
        window_size = int(fs*window_size_t)
        lead_in_data = tmp_data_res[1]['sig_name']
        print(lead_in_data)
        my_lead_all = []

        for tmp_lead in valid_lead:
            if tmp_lead in lead_in_data:
                my_lead_all.append(tmp_lead)
        if len(my_lead_all) != 0:
            for my_lead in range(1):
                pp_data = []
                pp_label = []
                channel = my_lead
                tmp_data = tmp_data_res[0][:, channel]
                # tmp_data = IIRRemoveBL(tmp_data, fs, Fc=0.67)
                idx_list = tmp_ann_res['sample']
                label_list = np.array(tmp_ann_res['symbol'])
                aux_list = np.array([i.strip('\x00') for i in tmp_ann_res['aux_note']])
                full_aux_list = [''] * tmp_data_res[1]['sig_len'] # expand aux to full length
                for i in range(len(aux_list)):
                    full_aux_list[idx_list[i]] = aux_list[i] # copy old aux
                    if label_list[i] in ['[', '!']:
                        full_aux_list[idx_list[i]] = '(VF' # copy VF start from beat labels
                    if label_list[i] in [']']:
                        full_aux_list[idx_list[i]] = '(N' # copy VF end from beat labels
                for i in range(1,len(full_aux_list)):
                    if full_aux_list[i] == '':
                        full_aux_list[i] = full_aux_list[i-1] # copy full_aux_list from itself, fill empty strings
                idx_start = 0
                flag = 0
                while idx_start < len(tmp_data) - window_size:
                    idx_end = idx_start+window_size

                    tmpdata = tmp_data[idx_start:idx_end]
                    if not -10 < np.mean(tmpdata) < 10 or np.std(tmpdata) == 0:
                        idx_start += stride_t * fs
                        continue
                    tmpdata = IIRRemoveBL(tmpdata, fs, Fc=0.67)
                    tmpdata = scipy.signal.resample(tmpdata, 128 * 10)
                    tmpdata = normalize_linear(tmpdata)
                    pp_data.append(tmpdata)
                    tmp_label_beat = label_list[np.logical_and(idx_list>=idx_start, idx_list<=idx_end)]
                    tmp_label_rhythm = full_aux_list[idx_start:idx_end] # be careful
                    tmp_label = list(np.unique(tmp_label_beat))+list(np.unique(tmp_label_rhythm))
                    tmp_label = get_label_map(tmp_label)
                    idx_start += stride_t * fs
                    if 'VF/VT' in tmp_label:
                        flag = 1
                        pp_label.append(2)
                    elif flag == 1:
                        pp_label.append(2)
                    else:
                        pp_label.append(1)
                count_1 = pp_label.count(1)
                pp_label = np.array(pp_label)
                if count_1 > 60:
                    pp_label[:count_1-60] = 0
                VF_PRE_DURATION.append(count_1)

                all_data["VF_" + record_name] = {"X": pp_data, "Y": pp_label}

    return all_data
def getVFData():
    window_size_t = 10  # second
    stride_t = 5  # second

    train_data = {}
    test_data = {}
    path ="/home/lzy/workspace/codeFile/VFearlywarning/sudden-cardiac-death-holter-database-1.0.0/"
    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')
    for record_name in all_record_name:
        if record_name in ["30", "31", "33", "34", "35", "36", "37", "38", "39", "41", "44", "45", "46", "47", "48", "50", "52"]:
            tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
            fs = tmp_data_res[1]['fs']
            window_size = int(fs * window_size_t)
            stride = int(fs * stride_t)
            file = open(path + record_name + '.hea')
            info = [int(i) for i in file.read().split()[-1].split(':')]
            time = (info[0] * 60 + info[1]) * 60 + info[2]
            time_list = [(time - 15*60, time + 5 * 60)]

            for i, (start_time, end_time) in enumerate(time_list):
                pp_data = []
                pp_label = []
                channel = 0
                tmp_data = tmp_data_res[0][:, channel][start_time * fs: end_time * fs]

                idx_start = 0
                while idx_start <= len(tmp_data) - window_size:
                    idx_end = idx_start + window_size

                    tmpdata = tmp_data[idx_start:idx_end]
                    if not -10 < np.mean(tmpdata) < 10 or np.std(tmpdata) == 0:
                        idx_start += stride
                        continue

                    tmpdata = IIRRemoveBL(tmpdata, fs, Fc=0.67)
                    tmpdata = scipy.signal.resample(tmpdata, 128 * 10)
                    tmpdata = normalize_linear(tmpdata)
                    pp_data.append(tmpdata)

                    if idx_end <= 10 * 60 * fs:
                        pp_label.append(0)
                    elif 10 * 60 * fs < idx_end <= 15 * 60 * fs:
                        pp_label.append(1)
                    else:
                        pp_label.append(2)
                    idx_start += stride

                train_data["VF_"+record_name] = {"X": pp_data, "Y": pp_label}
    path = '/home/lzy/workspace/codeFile/VFearlywarning/mit-bih-normal-sinus-rhythm-database-1.0.0'
    with open(os.path.join(path, 'RECORDS'), 'r') as fin:
        all_record_name = fin.read().strip().split('\n')
    for key in ["train", "test"]:
        if key == "train":
            nsr_list = all_record_name[:10]
        else:
            nsr_list = all_record_name[10:]
        for record_name in nsr_list:
            tmp_data_res = wfdb.rdsamp(path + '/' + record_name)
            fs = tmp_data_res[1]['fs']
            window_size = int(fs * window_size_t)
            stride = int(fs * stride_t)

            segment_time_stamp = random.sample(list(range(24)), 5)

            for time_stamp in segment_time_stamp:
                pp_data = []
                pp_label = []
                data_quality = []
                channel = 0
                idx_start = 0

                tmp_data = tmp_data_res[0][:, channel][(time_stamp * 60) * fs: (time_stamp * 60 + 20 * 60) * fs]

                tmp_data = IIRRemoveBL(tmp_data, fs, Fc=0.67)
                while idx_start <= len(tmp_data) - window_size:
                    idx_end = idx_start + window_size
                    tmpdata = scipy.signal.resample(tmp_data[idx_start:idx_end], 128 * 10)
                    if not -10 < np.mean(tmpdata) < 10 or np.std(tmpdata) == 0:
                        idx_start += stride
                        continue
                    tmpdata = normalize_linear(tmpdata)
                    pp_data.append(tmpdata)
                    pp_label.append(0)
                    idx_start += stride
                if key == "train":
                    train_data["NSR_{}_nsr_{}.pkl".format(record_name, time_stamp)] = {"X": pp_data, "Y": pp_label}
                else:
                    test_data["NSR_{}_nsr_{}.pkl".format(record_name, time_stamp)] = {"X": pp_data, "Y": pp_label}

    test_data.update(getCUDB())

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data4save = {"trainData": train_data, "testData": test_data, }
    np.save(save_path + "VF_all_data.npy", data4save)
    # with open(save_path + "VF_all_data.pkl", 'wb') as file:
    #     pickle.dump(data4save, file)

def extra_VFdata():
    dataset_dir = "/media/lzy/Elements SE/early_warning/VF_data"
    file_path = dataset_dir + "/VF_all_data.npy"

    file_data = np.load(file_path, allow_pickle=True).item()["testData"]

    file_data.update(getCUDB())
    data4save = {"trainData": np.load(file_path, allow_pickle=True).item()["trainData"], "testData": file_data, }
    np.save(save_path + "VF_all_data.npy", data4save)



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



def data_vis():

    # with open(save_path + "VF_all_data.pkl", 'rb') as file:
    for key in ["trainData", "testData"]:

        data_vf_train = np.load(save_path + "VF_all_data.npy", allow_pickle=True).item()["trainData"]

        for name in list(data_vf_train.keys()):
            data = data_vf_train[name]["X"]
            if not data:
                print(name)

    print(data_vf_train.keys())
    print(Counter(data_vf_train[list(data_vf_train.keys())[0]]["Y"]))
    data = data_vf_train[list(data_vf_train.keys())[0]]["X"]

    plt.figure()
    plt.plot(data[0], "r")

    plt.show()

if __name__ == "__main__":
    # getVFData()
    # data_vis()
    extra_VFdata()
    data_vf_train = np.load(save_path + "VF_all_data.npy", allow_pickle=True).item()["trainData"].keys()
    data_vf_test = np.load(save_path + "VF_all_data.npy", allow_pickle=True).item()["testData"].keys()

    print(len(data_vf_train))
    print(sum([1 for p in data_vf_train if "VF" in p ]))

    print(len(data_vf_test))

    print(sum([1 for p in data_vf_test if "VF" in p]))




















