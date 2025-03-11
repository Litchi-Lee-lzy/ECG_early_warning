import os
import pickle
from collections import Counter

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, \
    multilabel_confusion_matrix, matthews_corrcoef, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

from AF_task.PAF_warning_indicator_model import check_model_state_dict
from warning_model_utils.ablationSSL.ByolModel import CustomBYOL
from warning_model_utils.ablationSSL.cpcModel import cpcPretrainModel
from warning_model_utils.ablationSSL.simclrModel import CustomSimCLR
from warning_model_utils.pretrain_model import Config
from warning_model_utils.task2_comprison_model.ConvneXt import convnextv2_atto
from warning_model_utils.task2_comprison_model.fcn_wang import fcn_wang
from warning_model_utils.warning_indicator_model import indicator_cls_model
import seaborn as sns

dataset_dir = "/media/lzy/Elements SE/early_warning/VF_data/"
def seed_torch(seed=1029):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# data loader for warning indicator
class taskDataset(torch.utils.data.Dataset):

    def __init__(self, data_train=None, label_train=None, dataset_type=0):
        self.dataset_type = dataset_type

        print("data loading start!")

        # 现在，根据索引划分X, Y, Z数据
        self.dataSet = data_train
        self.label = label_train

        self.n_data = len(self.dataSet)


    def __getitem__(self, item):

        return self.dataSet[item], self.label[item]

    def __len__(self):
        return self.n_data



def training_one_epoch_model(model, dataloader, optimizer, epoch, max_epoch, model_name="trs"):
    loop = tqdm(enumerate(dataloader), desc="Training")
    loop.set_description(f'Epoch [{epoch}/{max_epoch}]')
    model.train()
    epoch_loss = 0
    for i, data_source in loop:
        optimizer.zero_grad()
        ecg, label = data_source[0].type(torch.FloatTensor).to(DEVICE), data_source[1].type(torch.FloatTensor).to(
            DEVICE)
        if model_name in ["cpc","byol","simclr"]:
            out_put = model.classify(ecg)
        else:
            out_put = model(ecg)

        loss = nn.CrossEntropyLoss()(out_put, label.long())
        loss.backward()
        loop.set_postfix(loss=loss.item())
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)



def updateClassCenter(model, dataloader):

    model.eval()
    class_feature_0 = np.empty((0, 512))
    class_feature_1 = np.empty((0, 512))
    class_feature_2 = np.empty((0, 512))

    for data_source in dataloader:
        ecg, label = data_source[0].type(torch.FloatTensor).to(DEVICE), data_source[1]
        label = label.numpy()

        out_put = model.getEmbedding(ecg)

        idx = np.where(label==0)[0]
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



def evaluate_model(model, dataloader, model_name="trs", show=False):
    model.eval()
    total_loss = 0.
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for data_source in dataloader:
            ecg, label = data_source[0].type(torch.FloatTensor).to(DEVICE), data_source[1].type(torch.FloatTensor).to(
                DEVICE)

            if model_name in ["cpc", "byol", "simclr"]:
                out_put = model.classify(ecg)
            else:
                out_put = model(ecg)
            _, predicted = torch.max(out_put.data, 1)
            true_labels += label.tolist()
            pred_labels += predicted.tolist()


        if show:
            f, ax = plt.subplots()
            cm = confusion_matrix(y_true=true_labels, y_pred=pred_labels)
            sns.heatmap(cm, annot=True, ax=ax)  # 画热力图

            ax.set_title('confusion matrix')  # 标题
            ax.set_xlabel('predict')  # x 轴
            ax.set_ylabel('true')  # y 轴
            # 命令行输出 混淆矩阵
            print('\nEvaluating....')
            print("TEST ACC:", accuracy_score(true_labels, pred_labels))
            print(classification_report(true_labels, pred_labels))
            print("Confusion Matrix:")
            print(cm)
            plt.show()

        return f1_score(true_labels, pred_labels, average='macro')


def ecg_vis(egc_sig, new_sig=None, title="ecg"):
    fig, ax = plt.subplots(figsize=(6, 2), dpi=200)

    ax.plot(np.arange(0, 10, 1 / 128), egc_sig, "k", alpha=0.99, lw=1.)
    if new_sig is not None:
        ax.plot(np.arange(0, 10, 1 / 128), new_sig, "r", alpha=0.99, lw=.6)
    ax.set_ylim(top=1.1, bottom=-.2)
    plt.title(title)
    # plt.axis("off")
    plt.show()

def getData(data_type ="trainData"):
    file_path = dataset_dir + "/VF_all_data.npy"
    data = np.empty((0, 1280))
    label = np.empty((0, 1))
    file_data = np.load(file_path, allow_pickle=True).item()[data_type]

    for file_name in tqdm(file_data.keys()):

        data_ = file_data[file_name]["X"]
        label_ = np.array(file_data[file_name]["Y"]).reshape(-1, 1)

        idx_abnormal = np.where(label_ > 0)[0]
        if len(idx_abnormal) > 0:
            data_ = np.array(data_)[idx_abnormal]
            label_ = np.array(label_)[idx_abnormal]
        elif "NSR" in file_name:
            data_ = np.array(data_)[:120]
            label_ = np.array(label_)[:120]

        patient_ecg = data_
        patient_label = label_

        data = np.concatenate((data, patient_ecg), axis=0)
        label = np.concatenate((label, patient_label), axis=0)

    label = label.reshape(-1)
    print(Counter(label))
    if data_type == "trainData":
        return train_test_split(data, label,test_size=0.2, random_state=42)
    else:
        return data, label

def get_model(model_name, num_classes=3):

    if model_name == "convnext":
        model = convnextv2_atto(num_classes=num_classes).to(DEVICE)

    elif model_name == "fcn_wang":
        model = fcn_wang(num_classes=num_classes).to(DEVICE)

    # self-supervised

    elif model_name == "trs":
        model = indicator_cls_model(num_classes=num_classes).to(DEVICE)

        pretrain_weight = torch.load(config.output_dir + f"mask_unmask_model_{100}.pth", map_location=DEVICE)
        model.load_state_dict(pretrain_weight, strict=False)
    elif model_name == "vit":
        model = indicator_cls_model(num_classes=num_classes).to(DEVICE)

    elif model_name == "cpc":
        output_dir = "/media/lzy/Elements SE/early_warning/pretrain_result/cpc/"
        model = cpcPretrainModel(num_classes=num_classes).to(DEVICE)
        pretrain_param = torch.load(output_dir + f"cpc_pretrain_39.pth", map_location=DEVICE)

        new_model_state_dict = model.state_dict()
        # 创建一个新的状态字典来保存更新后的参数
        new_model_state_dict = check_model_state_dict(new_model_state_dict, pretrain_param)
        model.load_state_dict(
            new_model_state_dict, strict=True)
        model_type = "self-supervised"
    elif model_name == "byol":
        output_dir = "/media/lzy/Elements SE/early_warning/pretrain_result/byol/"
        model = CustomBYOL(num_classes=num_classes).to(DEVICE)

        pretrain_param = torch.load(output_dir + f"byol_pretrain_81.pth",
                                    map_location=DEVICE)
        new_model_state_dict = model.state_dict()
        new_model_state_dict = check_model_state_dict(new_model_state_dict, pretrain_param)
        model.load_state_dict(
            new_model_state_dict, strict=True)
    elif model_name == "simclr":
        output_dir = "/media/lzy/Elements SE/early_warning/pretrain_result/simclr/"
        model = CustomSimCLR(num_classes=num_classes).to(DEVICE)

        pretrain_param = torch.load(output_dir + f"simclr_pretrain_96.pth",
                                    map_location=DEVICE)
        new_model_state_dict = model.state_dict()
        new_model_state_dict = check_model_state_dict(new_model_state_dict, pretrain_param)
        model.load_state_dict(
            new_model_state_dict, strict=True)

    else:
        print("wrong model name")
    return model

def _test_single_(model, dataload, model_name="trs"):
    result = {}
    import seaborn as sns
    def sensitivityCalc(Predictions, Labels):
        MCM = multilabel_confusion_matrix(Labels, Predictions,
                                          sample_weight=None,
                                          labels=None, samplewise=None)

        # 切片操作，获取每一个类别各自的 tn, fp, tp, fn
        tn_sum = MCM[:, 0, 0]  # True Negative
        fp_sum = MCM[:, 0, 1]  # False Positive

        tp_sum = MCM[:, 1, 1]  # True Positive
        fn_sum = MCM[:, 1, 0]  # False Negative

        # 这里加1e-6，防止 0/0的情况计算得到nan，即tp_sum和fn_sum同时为0的情况
        Condition_negative = tp_sum + fn_sum + 1e-6

        sensitivity = tp_sum / Condition_negative
        macro_sensitivity = np.average(sensitivity, weights=None)

        micro_sensitivity = np.sum(tp_sum) / np.sum(tp_sum + fn_sum)

        return macro_sensitivity, micro_sensitivity

    def specificityCalc(Predictions, Labels):
        MCM = multilabel_confusion_matrix(Labels, Predictions,
                                          sample_weight=None,
                                          labels=None, samplewise=None)
        tn_sum = MCM[:, 0, 0]
        fp_sum = MCM[:, 0, 1]

        tp_sum = MCM[:, 1, 1]
        fn_sum = MCM[:, 1, 0]

        Condition_negative = tn_sum + fp_sum + 1e-6

        Specificity = tn_sum / Condition_negative
        macro_specificity = np.average(Specificity, weights=None)

        micro_specificity = np.sum(tn_sum) / np.sum(tn_sum + fp_sum)

        return macro_specificity, micro_specificity

    model.eval()
    tru = []
    preds = []
    logits = []
    for data, label in dataload:
        # traces = traces.transpose(1, 2)
        data, label = data.float().to(DEVICE), label.to(DEVICE)
        with torch.no_grad():
            # Reinitialize grad
            model.zero_grad()
            # Send to device
            # Forward pass
            if model_name in ["cpc", "byol", "simclr"]:
                logit = model.classify(data)
            else:
                logit = model.forward(data)

            _, pred = torch.max(logit, 1)
            truth = label
            pred = pred.cpu()
            logits.append(logit.cpu().tolist())
            preds.append(pred.tolist())
            tru.append(truth.cpu().tolist())

    preds_flat = [item for pred in preds for item in pred]
    truth_flat = [item for truth in tru for item in truth]
    logit_flat = [item for logit in logits for item in logit]

    # np.save(f"./modelResult/baseline_hospital/{model_name}_ori_rs.npy", {"logit": logit_flat, "label": truth_flat})
    # 命令行输出 混淆矩阵
    print('\nEvaluating....')
    print("TEST ACC:", accuracy_score(truth_flat, preds_flat))
    print("TEST MCC:", matthews_corrcoef(truth_flat, preds_flat))
    macro_specificity, micro_specificity = specificityCalc(preds_flat, truth_flat)
    macro_sensitivity, micro_sensitivity = sensitivityCalc(preds_flat, truth_flat)
    print("TEST SPE:", macro_specificity)
    print("TEST SEN:", macro_sensitivity)
    print("TEST F1:", f1_score(truth_flat, preds_flat, average='macro'))
    # 计算macro AUC
    auc_scores = []
    for i in range(len(logit_flat[0])):
        # 将当前类别作为正样本
        y_true_binary = [1 if label == i else 0 for label in truth_flat]
        # 计算AUC
        auc = roc_auc_score(y_true_binary, [pred[i] for pred in logit_flat])
        auc_scores.append(auc)

    # 计算平均AUC
    macro_auc = sum(auc_scores) / len(auc_scores)
    result["acc"] = round(accuracy_score(truth_flat, preds_flat) * 100, 2)
    result["spe"] = round(macro_specificity * 100, 2)
    result["sen"] = round(macro_sensitivity * 100, 2)
    result["f1"] = round(f1_score(truth_flat, preds_flat, average='macro') * 100, 2)
    result["mcc"] = round(matthews_corrcoef(truth_flat, preds_flat) * 100, 2)

    return result

if __name__ == "__main__":
    config = Config()
    DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    vt_model_save_path = "/media/lzy/Elements SE/early_warning/VF_model/classification_model_checkpoint/"
    finetune_epoch = 10
    flag = 0 # 0 正常区域聚类 1 训练分类模型 AF PRE-AF N 2 构建LSTM模型

    if flag == 0:

        data_train, data_val, label_train, label_val = getData("trainData")
        dataloader = DataLoader(taskDataset(data_train=data_train, label_train=label_train), batch_size=128, shuffle=True)
        dataloader_val = DataLoader(taskDataset(data_train=data_val, label_train=label_val), batch_size=256,
                                    shuffle=False)
        data_test, label_test = getData("testData")
        dataloader_test = DataLoader(taskDataset(data_train=data_test, label_train=label_test), batch_size=256,
                                     shuffle=False)
        model_list = ["fcn_wang", "convnext", "vit", "trs", "cpc", "byol", "simclr"]

        # model_name = "trs"
        all_result = {}
        for model_name in model_list:
            print(f"========================={model_name}=========================")

            model = get_model(model_name)
            model_save_path = vt_model_save_path + "{}_status_dis_model.pth".format(model_name)
            if os.path.exists(model_save_path):
                tqdm.write("best model loading...")
                best_mode_param = torch.load(model_save_path, map_location=DEVICE)
                model.load_state_dict(best_mode_param, strict=True)
            else:
                if model_name == "trs":
                    low_lr = 1e-5  # 预训练模块的学习率
                    high_lr = 1e-3  # 没有预训练权重的模块的学习率
                    pretrained_params = list(model.encoder.parameters()) + \
                                        list(model.to_patch.parameters()) + list(model.patch_to_emb.parameters())
                    pretrained_param_ids = set(id(p) for p in pretrained_params)
                    other_params = [p for p in model.parameters() if id(p) not in pretrained_param_ids]
                    optimizer = torch.optim.AdamW([
                        {'params': pretrained_params, 'lr': low_lr},
                        {'params': other_params, 'lr': high_lr}
                    ], weight_decay=1e-6)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epoch,
                                                                       eta_min=config.min_lr)
                val_loss_current = 0
                for epoch in range(1, finetune_epoch + 1):
                    training_one_epoch_model(model=model, dataloader=dataloader,
                                             optimizer=optimizer, epoch=epoch,
                                             max_epoch=finetune_epoch, model_name=model_name)
                    scheduler.step()
                    total_loss = evaluate_model(model, dataloader_val, model_name=model_name)
                    if total_loss > val_loss_current:
                        val_loss_current = total_loss
                        # updateClassCenter(model, dataloader, dataloader_test)
                        if not os.path.exists(vt_model_save_path):
                            os.makedirs(vt_model_save_path)
                        print('best_performance: {:.4f}'.format(total_loss))
                        torch.save(model.state_dict(),
                                   vt_model_save_path + "{}_status_dis_model.pth".format(model_name))

            result = _test_single_(model, dataloader_test, model_name=model_name)

            all_result[model_name] = result
        df = pd.DataFrame(all_result)
        # 将DataFrame保存为CSV文件
        csv_file = os.path.join(vt_model_save_path, 'vf_classification_result.csv')
        df.to_csv(csv_file, index=True)  # index=False以避免将行索引写入CSV


