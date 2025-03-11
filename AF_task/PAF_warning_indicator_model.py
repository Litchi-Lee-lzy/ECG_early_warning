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
import seaborn as sns
from warning_model_utils.ablationSSL.ByolModel import CustomBYOL
from warning_model_utils.ablationSSL.cpcModel import cpcPretrainModel
from warning_model_utils.ablationSSL.simclrModel import CustomSimCLR
from warning_model_utils.pretrain_model import Config
from warning_model_utils.task2_comprison_model.ConvneXt import convnextv2_atto
from warning_model_utils.task2_comprison_model.fcn_wang import fcn_wang
from warning_model_utils.warning_indicator_model import indicator_cls_model

def seed_torch(seed=1029):
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def GenerateTrainAndTest():
    dataset_dir = "/media/lzy/Elements SE/early_warning/PAF_data/2.0/paf_warning_dataset/"

    dataset_files = os.listdir(dataset_dir)
    dataset_files = [filename for filename in dataset_files if filename.endswith('.pkl')]

    if "train_test_spilt.pkl" in dataset_files:
        with open(dataset_dir + "train_test_spilt.pkl", 'rb') as file:
            patient = pickle.load(file)
            return patient["trainPatient"], patient["valPatient"], patient["testPatient"]

    seed_torch(123)

    trainData, testData = train_test_split(dataset_files, test_size=0.3, random_state=111)
    trainData, valData = train_test_split(trainData, test_size=0.2, random_state=111)
    with open(dataset_dir + "train_test_spilt.pkl", 'wb') as file:
        pickle.dump({"trainPatient": trainData, "valPatient": valData, "testPatient": testData}, file)
    return trainData, valData, testData


# data loader for ECG STATE model
class taskDataset(torch.utils.data.Dataset):

    def __init__(self, file_list=None, task="sta_dis", train=0):
        dataset_dir = "./AF_data/"
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
                idx_abnormal = np.where(label_ > 0)[0]
                data_ = np.array(data_)[idx_abnormal]
                label_ = np.array(label_)[idx_abnormal]
            patient_ecg = []
            patient_label = []

            for (e, l) in zip(data_, label_):
                patient_ecg.append(e)
                patient_label.append(l)
            data = np.concatenate((data, patient_ecg), axis=0)
            label = np.concatenate((label, patient_label), axis=0)

        label = label.reshape(-1)
        self.data_status = train
        print(Counter(label))
        self.dataSet = torch.from_numpy(data)
        self.label = torch.from_numpy(label)
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


def updateClassCenter(model, dataloader, dataloader_test):

    model.eval()
    class_feature_0 = np.empty((0, 512))
    class_feature_1 = np.empty((0, 512))
    class_feature_2 = np.empty((0, 512))
    for dataloader in [dataloader, dataloader_test]:
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

def _test_(preds, tru, logits):

    preds_flat = [pred for pred in preds]
    truth_flat = [truth for truth in tru]
    logit_flat = [item for logit in logits for item in logit]
    cm = confusion_matrix(y_true=truth_flat, y_pred=preds_flat)
    # 命令行输出 混淆矩阵
    print('\nEvaluating....')
    print("TEST ACC:", accuracy_score(truth_flat, preds_flat))
    print("TEST MCC:", matthews_corrcoef(truth_flat, preds_flat))
    macro_specificity, micro_specificity = specificityCalc(preds_flat, truth_flat)
    macro_sensitivity, micro_sensitivity = sensitivityCalc(preds_flat, truth_flat)
    print("TEST SPE:", macro_specificity)
    print("TEST SEN:", macro_sensitivity)
    print("TEST F1:", f1_score(truth_flat, preds_flat, average='macro'))

    plt.figure(1)
    x_tick = ["Normal", 'Pre', "AF"]
    y_tick = ["Normal", 'Pre', "AF"]
    plt.subplots(figsize=(2, 2), dpi=200)
    plt.tick_params(bottom=False, top=False, left=False, right=False)
    plt.grid = False
    cm = confusion_matrix(y_true=truth_flat, y_pred=preds_flat)
    sns.heatmap(cm, cmap='Blues', fmt='g', annot=True, cbar=False,
                xticklabels=x_tick, yticklabels=y_tick)
    plt.show()

    return preds_flat, truth_flat,

def check_model_state_dict(new_weight, pretrain_weight):
    updated_state_dict = {}
    # 遍历预训练的参数
    for name, param in pretrain_weight.items():
        if name in new_weight:
            if new_weight[name].shape == param.shape:
                # 如果形状匹配，则加载参数
                updated_state_dict[name] = param
            else:
                # 如果形状不匹配，则跳过该参数
                print(f'Skip loading parameter: {name}; '
                      f'Shape mismatch (Pretrained: {param.shape}, New: {new_weight[name].shape})')
        else:
            # 如果新模型没有这个参数，则跳过
            print(f'Skip loading parameter: {name}; '
                  'It does not exist in the new model.')

    # 更新新模型的 state_dict
    new_weight.update(updated_state_dict)
    return new_weight


def get_model(model_name, num_classes=3):
    # supervised
    model_type = "supervised"
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
        model_type = "self-supervised"
    elif model_name == "simclr":
        output_dir = "/media/lzy/Elements SE/early_warning/pretrain_result/simclr/"
        model = CustomSimCLR(num_classes=num_classes).to(DEVICE)

        pretrain_param = torch.load(output_dir + f"simclr_pretrain_96.pth",
                                    map_location=DEVICE)
        new_model_state_dict = model.state_dict()
        new_model_state_dict = check_model_state_dict(new_model_state_dict, pretrain_param)
        model.load_state_dict(
            new_model_state_dict, strict=True)
        model_type = "self-supervised"

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
    paf_model_save_path = "./AF_model/"
    finetune_epoch = 10
    save_path = paf_model_save_path + "classification_model_checkpoint/"

    trainData, valData, testData = GenerateTrainAndTest()
    dataloader = DataLoader(taskDataset(file_list=trainData, task="sta_dis", train=2), batch_size=128, shuffle=True)
    dataloader_val = DataLoader(taskDataset(file_list=valData, task="sta_dis", train=2), batch_size=256,
                                 shuffle=False)
    dataloader_test = DataLoader(taskDataset(file_list=testData, task="sta_dis", train=2), batch_size=256, shuffle=False)
    model_list = [ "fcn_wang", "convnext", "vit", "trs", "cpc", "byol", "simclr"]

    all_result = {}
    for model_name in model_list:
        print(f"========================={model_name}=========================")
        model = get_model(model_name)
        model_save_path = save_path + "{}_status_dis_model.pth".format(model_name)
        if os.path.exists(model_save_path):
            tqdm.write("best model loading...")
            best_mode_param = torch.load(model_save_path, map_location=DEVICE)
            model.load_state_dict(best_mode_param, strict=True)
            model.eval()
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
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    print('best_performance: {:.4f}'.format(total_loss))
                    torch.save(model.state_dict(),
                               save_path + "{}_status_dis_model.pth".format(model_name))


        result = _test_single_(model, dataloader_test, model_name=model_name)

        all_result[model_name] = result
    df = pd.DataFrame(all_result)
    # 将DataFrame保存为CSV文件
    csv_file = os.path.join(save_path, 'paf_classification_result.csv')
    df.to_csv(csv_file, index=True)  # index=False以避免将行索引写入CSV





