import argparse
import os
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from dataloader import MyDataset, collate_fn
from functools import partial
from torch.utils.data import DataLoader, Subset, random_split
import torch.optim as optim

# 单模态的使用
# from model_feature_Davis import MdDTI #仅添加药物文本特征
# from model_image_Davis import MdDTI #仅添加药物图像
from model_tss_Davis import MdDTI #仅添加靶标二级结构信息



# from model import MdDTI  # 复现原文模型
# from model_feature_image_tss_Davis import MdDTI #添加药物的图像和文本特征和靶标二级结构信息

#消融实验
# from model_without_image_Davis import MdDTI #仅移除药物图像信息
# from model_without_text_Davis import MdDTI #仅移除药物化学文本信息
# from model_without_tss_Davis import MdDTI #仅移除靶标二级结构信息
# from model_without_GAT_Davis import MdDTI #仅移除药物的GAT模块
# from model_without_loss_function_Davis import MdDTI #仅移除一致性损失函数
# from model_without_feature_decoder_module_Davis import MdDTI #仅移除feature decoder module模块

from utils import load_common_data, load_train_data_DTI, get_test_idx, get_five_fold_idx
from evaluate import evaluate_DTI, save_result_DTI, save
from time import time
import os
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

l_drug_dict = len(pickle.load(open('./common/Davis/dict/atom_r_dict', 'rb')))
l_ICMF_dict = len(pickle.load(open('./common/Davis/dict/icmf_dict', 'rb')))
l_target_dict = len(pickle.load(open('./common/Davis/dict/target_dict', 'rb')))

parser = argparse.ArgumentParser()
parser.add_argument('--substructure', default='ICMF', help='Select the drug 2D substructure. (ICMF or ESPF)')
parser.add_argument('--device', default='cuda:0', help='Disables CUDA training.')
parser.add_argument('--batch-size', type=int, default=16, help='Number of batch_size')
parser.add_argument('--max-target', type=int, default=1000, help='The maximum length of the target.')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Initial learning rate. (human and celegans is 1e-4. Davis and KIBA is 5e-4)')
parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr-step-size', type=int, default=15, help='Period of learning rate decay.')
parser.add_argument('--lr-gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay.')
parser.add_argument('--count-decline', default=30, help='Early stoping.')
parser.add_argument('--task', default='DTI', help='Train DTI datasets.')
parser.add_argument('--drug-emb-dim', type=int, default=64, help='The embedding dimension of drug')
parser.add_argument('--target-emb-dim', type=int, default=64, help='The embedding dimension of target')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')  # 120
parser.add_argument('--p', type=int, default=10, help='The scale factor of coordinates.')
parser.add_argument('--emb1_size', type=int, default=256, help='多头注意力机制的嵌入向量')
parser.add_argument('--emb2_size', type=int, default=64, help='多头注意力机制的嵌入向量')
# parser.add_argument('--K-FOLD', type=int, default=2, help='The 5 cross validation.')

args = parser.parse_args()
if args.device == 'cuda:0':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print("cuda is not available!!!")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
print('>>>>>> The code run on the ', device)
criterion = nn.CrossEntropyLoss()


def show_result(Accuracy_List,Precision_List,Recall_List,AUC_List,AUPR_List):
    Accuracy_mean, Accuracy_std = np.mean(Accuracy_List), np.std(Accuracy_List)
    Precision_mean, Precision_std = np.mean(Precision_List), np.std(Precision_List)
    Recall_mean, Recall_std = np.mean(Recall_List), np.std(Recall_List)
    AUC_mean, AUC_std = np.mean(AUC_List), np.std(AUC_List)
    PRC_mean, PRC_std = np.mean(AUPR_List), np.std(AUPR_List)
    # with open("./实验结果/Davis/添加药物图像和文本特征和靶标的二级结构信息/加入标准差的实验结果/Davis_results(std).txt", 'w') as f:
    # with open("./实验结果/Davis/单模态的使用/仅添加药物文本特征/Davis_text(std).txt", 'w') as f:
    # with open("./实验结果/Davis/单模态的使用/仅添加药物图像/Davis_image(std).txt", 'w') as f:
    with open("./实验结果/Davis/单模态的使用/仅添加靶标二级结构信息/Davis_tss(std).txt", 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_std) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_std) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std) + '\n')
        f.write('AUPR(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std) + '\n')

    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_std))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_std))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std))
    print('AUPR(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std))



def train(model, optimizer, data_loader):
    model.train()
    pred_labels = []
    predictions = []
    all_labels = []
    total_loss = []
    for target, drug, labels in tqdm(data_loader):
        optimizer.zero_grad()

        # out, loss2 = model(drug, target)  # 不添加模块的代码
        # out, loss2, rate1, rate2, rate3= model(drug, target)  # 添加模块的代码
        out, loss2, rate1, rate2 = model(drug, target)  # 添加模块的代码
        # out, rate1, rate2 = model(drug, target)  # 仅移除一致性损失函数

        labels = torch.FloatTensor(labels).to(args.device)
        loss = criterion(out.float(), labels.reshape(labels.shape[0]).long()) + loss2
        # loss = criterion(out.float(), labels.reshape(labels.shape[0]).long()) # 仅移除一致性损失函数
        total_loss.append(loss.cpu().detach())
        ys = F.softmax(out, 1).to('cpu').data.numpy()
        pred_labels += list(map(lambda x: np.argmax(x), ys))  # (32)预测结果
        predictions += list(map(lambda x: x[1], ys))  # (32)标签为1的值
        all_labels += labels.cpu().numpy().reshape(-1).tolist()

        loss.backward()
        optimizer.step()
    res = evaluate_DTI(total_loss, pred_labels, predictions, all_labels)
    return res


def test(model, data_loader):
    model.eval()
    pred_labels = []
    predictions = []
    all_labels = []
    total_loss = []

    with torch.no_grad():
        for target, drug, labels in tqdm(data_loader, colour='blue'):

            # out, loss2 = model(drug, target)  # 不添加模块的代码
            # out, loss2, rate1, rate2, rate3 = model(drug, target)  # 添加模块的代码
            out, loss2, rate1, rate2 = model(drug, target)  # 添加模块的代码
            # out, rate1, rate2 = model(drug, target)  # 仅移除一致性损失函数

            labels = torch.FloatTensor(labels).to(args.device)
            loss = criterion(out.float(), labels.reshape(labels.shape[0]).long()) + loss2
            # loss = criterion(out.float(), labels.reshape(labels.shape[0]).long())  # 仅移除一致性损失函数
            total_loss.append(loss.cpu().detach())
            ys = F.softmax(out, 1).to('cpu').data.numpy()
            pred_labels += list(map(lambda x: np.argmax(x), ys))
            predictions += list(map(lambda x: x[1], ys))
            all_labels += labels.cpu().numpy().reshape(-1).tolist()

    res = evaluate_DTI(total_loss, pred_labels, predictions, all_labels)
    return res

'''
# 改进代码  加入5折交叉验证
def run(model, optimizer, train_loader, valid_loader, test_loader, dataset, fold_i):
    os.makedirs('./save_model/{}/'.format(dataset), exist_ok=True)
    model_path = './save_model/model.pth'
    if os.path.exists(model_path):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print(">>> Continue training ...")

    valid_max_AUC = 0.5
    decline = 0

    best_auc, best_aupr = 0.4, 0.5

    best_auc_fpr = []
    best_auc_tpr = []
    best_aupr_precision = []
    best_aupr_recall = []

    # 定义是否是第一次迭代的标志
    is_first_iteration = True

    for epoch in range(1, args.epochs + 1):
        # print('****** epoch:{} ******'.format(epoch))
        print('******Fold:{}-epoch:{}******'.format(fold_i + 1, epoch))
        res_train = train(model, optimizer, train_loader)
        save_result_DTI(dataset, "Train", res_train, epoch)

        res_valid = test(model, valid_loader)
        save_result_DTI(dataset, "Valid", res_valid, epoch)

        res_test = test(model, test_loader)
        save_result_DTI(dataset, "Test", res_test, epoch)

        # 提取前六列数据
        columns_to_save = ['Acc', 'Pre', 'Recall', 'AUC', 'AUPR', 'LOSS']
        data_to_save = res_test[:6]

        # 创建 DataFrame
        df_to_save = pd.DataFrame([data_to_save], columns=columns_to_save)

        # 保存 DataFrame 为 CSV 文件，追加数据，不写入列名
        # csv_path = './实验结果/Davis/仅移除药物图像信息/Davis_without_image_2.csv'
        # csv_path = './实验结果/Davis/仅移除药物化学文本信息/Davis_without_text_1.csv'
        # csv_path = './实验结果/Davis/仅移除靶标二级结构信息/Davis_without_tss.csv'
        csv_path = './实验结果/Davis/添加药物图像和文本特征和靶标的二级结构信息/Davis_result_std.csv'
        # 写入列名（第一次迭代）或追加数据（后续迭代）
        if is_first_iteration:
            df_to_save.to_csv(csv_path, index=False, mode='a', header=True)
            is_first_iteration = False
        else:
            df_to_save.to_csv(csv_path, index=False, mode='a', header=False)

        # 检查当前 AUC 是否是最大值
        if res_test[3] > best_auc:
            best_auc = res_test[3]
            best_auc_fpr, best_auc_tpr = res_test[6], res_test[7]
            # 保存到 auc_fpr_tpr.csv 文件
            df_auc_fpr_tpr = pd.DataFrame({'fpr': best_auc_fpr, 'tpr': best_auc_tpr})
            # df_auc_fpr_tpr.to_csv('./实验结果/Davis/仅移除药物图像信息/Davis_without_image_auc_fpr_tpr_2.csv', index=False)
            # df_auc_fpr_tpr.to_csv('./实验结果/Davis/仅移除药物化学文本信息/Davis_without_text_auc_fpr_tpr_1.csv', index=False)
            # df_auc_fpr_tpr.to_csv('./实验结果/Davis/仅移除靶标二级结构信息/Davis_without_tss_auc_fpr_tpr.csv', index=False)
            df_auc_fpr_tpr.to_csv('./实验结果/Davis/添加药物图像和文本特征和靶标的二级结构信息/Davis_auc_fpr_tpr_std.csv', index=False)

        # 检查当前 AUPR 是否是最大值
        if res_test[4] > best_aupr:
            best_aupr = res_test[4]
            best_aupr_recall, best_aupr_precision = res_test[8], res_test[9]
            # 保存到 aupr_precision_recall.csv 文件
            df_aupr_precision_recall = pd.DataFrame({'recall': best_aupr_recall, 'precision': best_aupr_precision})
            # df_aupr_precision_recall.to_csv('./实验结果/Davis/仅移除药物图像信息/Davis_without_image_aupr_recall_precision_2.csv', index=False)
            # df_aupr_precision_recall.to_csv('./实验结果/Davis/仅移除药物化学文本信息/Davis_without_text_aupr_recall_precision_1.csv',index=False)
            # df_aupr_precision_recall.to_csv('./实验结果/Davis/仅移除靶标二级结构信息/Davis_without_tss_aupr_recall_precision.csv',index=False)
            df_aupr_precision_recall.to_csv('./实验结果/Davis/添加药物图像和文本特征和靶标的二级结构信息/Davis_aupr_recall_precision_std.csv',index=False)

        if res_valid[3] > valid_max_AUC:
            valid_max_AUC = res_valid[3]
            decline = 0
            save_path = './save_model/{}/{}_valid_best_checkpoint_{}.pth'.format(dataset, args.substructure, epoch)
            torch.save(model.state_dict(), save_path)
        else:
            decline = decline + 1
            if decline >= args.count_decline:
                print("EarlyStopping !!!")
                break

        if epoch % args.lr_step_size == 0:  # 每隔n次学习率衰减一次
            optimizer.param_groups[0]['lr'] *= args.lr_gamma
'''




# def save_data(model,test_loader):
#     model_path = './save_model/human/ICMF_valid_best_checkpoint_39.pth'
#     state_dict = torch.load(model_path)
#     model.load_state_dict(state_dict)
#     res_test = test(model, test_loader)
#     print("xxx")

s = time()

if __name__ == '__main__':
    #############################################################
    # Davis and KIBA: lr=5e-4, lr_step_size: 15, epochs: 130    #
    #############################################################
    datasets = ['Davis']
    common_data = load_common_data(args.substructure)
    if args.substructure.upper() == 'ICMF':
        l_substructures_dict = l_ICMF_dict
    else:
        l_substructures_dict = 23533

    for dataset in datasets:
        train_data = load_train_data_DTI(dataset)
        train_data_iter = MyDataset(common_data, train_data, args)

        '''
        # 原代码
        train_idx, test_idx = get_test_idx(len(train_data_iter), 0.2, args.batch_size)
        train_dataset_iter = Subset(train_data_iter, train_idx)
        test_iter = Subset(train_data_iter, test_idx)

        train_idx, valid_idx = get_test_idx(len(train_dataset_iter), 0.1, args.batch_size)
        train_iter = Subset(train_dataset_iter, train_idx)
        valid_iter = Subset(train_dataset_iter, valid_idx)
        # print("train_iter: ", len(train_iter))
        # print("test_iter: ", len(test_iter))
        # print("valid_iter: ", len(valid_iter))
        '''

        Acc_List_stable = []
        Pre_List_stable = []
        Recall_List_stable = []
        AUC_List_stable = []
        AUPR_List_stable = []

        # 改进的代码1
        # 从总的数据长度中选取0.1作为测试集 剩余的作为训练集
        # 然后再从训练集中再选取0.1作为验证集 剩余的作为训练集
        # 因此 训练集：验证集：测试集 约等于 8：1：1
        train_idx, test_idx = get_test_idx(len(train_data_iter), 0.1, args.batch_size)
        train_dataset_iter = Subset(train_data_iter, train_idx)
        # print("train_dataset_iter: ", len(train_dataset_iter)) #
        test_iter = Subset(train_data_iter, test_idx)
        # print("test_iter: ", len(test_iter)) #

        train_idx, valid_idx = get_test_idx(len(train_dataset_iter), 0.1, args.batch_size)
        train_iter = Subset(train_dataset_iter, train_idx)
        valid_iter = Subset(train_dataset_iter, valid_idx)
        # print("train_iter: ", len(train_iter)) #
        # print("valid_iter: ", len(valid_iter)) #

        my_collate_fn = partial(collate_fn, device=args.device)
        train_loader = DataLoader(train_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)
        valid_loader = DataLoader(valid_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)
        test_loader = DataLoader(test_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)

        model = MdDTI(l_drug_dict, l_target_dict, l_substructures_dict, args)
        model.to(args.device)

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


        # run(model, optimizer, train_loader, valid_loader, test_loader, dataset, fold_i)
        os.makedirs('./save_model/{}/'.format(dataset), exist_ok=True)
        model_path = './save_model/model.pth'
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            print(">>> Continue training ...")

        valid_max_AUC = 0.5
        decline = 0

        best_auc, best_aupr = 0.4, 0.5

        # 定义是否是第一次迭代的标志
        is_first_iteration = True

        # 创建空列表来收集每个折的评估结果
        all_res_test = []

        for epoch in range(1, args.epochs + 1):
            print('****** epoch:{} ******'.format(epoch))
            res_train = train(model, optimizer, train_loader)
            save_result_DTI(dataset, "Train", res_train, epoch)

            res_valid = test(model, valid_loader)
            save_result_DTI(dataset, "Valid", res_valid, epoch)

            res_test = test(model, test_loader)
            all_res_test.append(res_test)
            save_result_DTI(dataset, "Test", res_test, epoch)

            # 提取前六列数据
            columns_to_save = ['Acc', 'Pre', 'Recall', 'AUC', 'AUPR', 'LOSS']
            data_to_save = res_test[:6]


            # 创建 DataFrame
            df_to_save = pd.DataFrame([data_to_save], columns=columns_to_save)

            # 保存 DataFrame 为 CSV 文件，追加数据，不写入列名
            # csv_path = './实验结果/Davis/添加药物图像和文本特征和靶标的二级结构信息/加入标准差的实验结果/Davis_result_std.csv'
            # csv_path = './实验结果/Davis/单模态的使用/仅添加药物文本特征/Davis_text.csv'
            # csv_path = './实验结果/Davis/单模态的使用/仅添加药物图像/Davis_image.csv'
            csv_path = './实验结果/Davis/单模态的使用/仅添加靶标二级结构信息/Davis_tss.csv'
            # 写入列名（第一次迭代）或追加数据（后续迭代）
            if is_first_iteration:
                df_to_save.to_csv(csv_path, index=False, mode='a', header=True)
                is_first_iteration = False
            else:
                df_to_save.to_csv(csv_path, index=False, mode='a', header=False)

            # 检查当前 AUC 是否是最大值
            if res_test[3] > best_auc:
                best_auc = res_test[3]
                best_auc_fpr, best_auc_tpr = res_test[6], res_test[7]
                # 保存到 auc_fpr_tpr.csv 文件
                df_auc_fpr_tpr = pd.DataFrame({'fpr': best_auc_fpr, 'tpr': best_auc_tpr})
                # df_auc_fpr_tpr.to_csv('./实验结果/Davis/添加药物图像和文本特征和靶标的二级结构信息/加入标准差的实验结果/Davis_auc_fpr_tpr_std.csv', index=False)
                # df_auc_fpr_tpr.to_csv('./实验结果/Davis/单模态的使用/仅添加药物文本特征/Davis_text_auc_fpr_tpr.csv',index=False)
                # df_auc_fpr_tpr.to_csv('./实验结果/Davis/单模态的使用/仅添加药物图像/Davis_image_auc_fpr_tpr.csv', index=False)
                df_auc_fpr_tpr.to_csv('./实验结果/Davis/单模态的使用/仅添加靶标二级结构信息/Davis_tss_auc_fpr_tpr.csv', index=False)

            # 检查当前 AUPR 是否是最大值
            if res_test[4] > best_aupr:
                best_aupr = res_test[4]
                best_aupr_recall, best_aupr_precision = res_test[8], res_test[9]
                # 保存到 aupr_precision_recall.csv 文件
                df_aupr_precision_recall = pd.DataFrame(
                    {'recall': best_aupr_recall, 'precision': best_aupr_precision})
                # df_aupr_precision_recall.to_csv('./实验结果/Davis/添加药物图像和文本特征和靶标的二级结构信息/加入标准差的实验结果/Davis_aupr_recall_precision_std.csv', index=False)
                # df_aupr_precision_recall.to_csv('./实验结果/Davis/单模态的使用/仅添加药物文本特征/Davis_text_aupr_recall_precision.csv', index=False)
                # df_aupr_precision_recall.to_csv('./实验结果/Davis/单模态的使用/仅添加药物图像/Davis_image_aupr_recall_precision.csv', index=False)
                df_aupr_precision_recall.to_csv('./实验结果/Davis/单模态的使用/仅添加靶标二级结构信息/Davis_tss_aupr_recall_precision.csv', index=False)

            if res_valid[3] > valid_max_AUC:
                valid_max_AUC = res_valid[3]
                decline = 0
                # save_path = './save_model/{}/{}_valid_best_checkpoint_{}.pth'.format(dataset, args.substructure,epoch)
                # torch.save(model.state_dict(), save_path)
            else:
                decline = decline + 1
                if decline >= args.count_decline:
                    print("EarlyStopping !!!")
                    break

            if epoch % args.lr_step_size == 0:  # 每隔n次学习率衰减一次
                optimizer.param_groups[0]['lr'] *= args.lr_gamma

        for res in all_res_test:
            Acc = res[0]
            Acc_List_stable.append(Acc)

            Pre = res[1]
            Pre_List_stable.append(Pre)

            Recall = res[2]
            Recall_List_stable.append(Recall)

            AUC = res[3]
            AUC_List_stable.append(AUC)

            AUPR = res[4]
            AUPR_List_stable.append(AUPR)


        show_result(Acc_List_stable, Pre_List_stable, Recall_List_stable, AUC_List_stable, AUPR_List_stable)



# 运行时间的计算
e = time()
elapsed_time_seconds = e - s

# 将秒转换成分钟
elapsed_time_minutes = elapsed_time_seconds / 60
print(f"测试运行时间: {elapsed_time_minutes:.2f} minutes")



'''
# 改进的代码2 交叉验证 下面的代码是可以跑通的，但是我不做交叉验证 
        for fold_i in range(1, args.K_FOLD + 1):

            train_idx, test_idx = get_test_idx(len(train_data_iter), 0.1, args.batch_size)
            train_dataset_iter = Subset(train_data_iter, train_idx)
            print("train_dataset_iter: ", len(train_dataset_iter)) #
            test_iter = Subset(train_data_iter, test_idx)
            print("test_iter: ", len(test_iter)) #

            train_idx, valid_idx = get_test_idx(len(train_dataset_iter), 0.1, args.batch_size)
            train_iter = Subset(train_dataset_iter, train_idx)
            valid_iter = Subset(train_dataset_iter, valid_idx)
            print("train_iter: ", len(train_iter)) #
            print("valid_iter: ", len(valid_iter)) #       

            my_collate_fn = partial(collate_fn, device=args.device)
            train_loader = DataLoader(train_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)
            valid_loader = DataLoader(valid_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)
            test_loader = DataLoader(test_iter, batch_size=args.batch_size, collate_fn=my_collate_fn, drop_last=True)

            model = MdDTI(l_drug_dict, l_target_dict, l_substructures_dict, args)
            model.to(args.device)

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


            # run(model, optimizer, train_loader, valid_loader, test_loader, dataset, fold_i)
            os.makedirs('./save_model/{}/'.format(dataset), exist_ok=True)
            model_path = './save_model/model.pth'
            if os.path.exists(model_path):
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)
                print(">>> Continue training ...")

            valid_max_AUC = 0.5
            decline = 0

            best_auc, best_aupr = 0.4, 0.5

            # 定义是否是第一次迭代的标志
            is_first_iteration = True

            # 创建空列表来收集每个折的评估结果
            all_res_test = []

            for epoch in range(1, args.epochs + 1):
                # print('****** epoch:{} ******'.format(epoch))
                print('******Fold:{}-epoch:{}******'.format(fold_i, epoch))
                res_train = train(model, optimizer, train_loader)
                save_result_DTI(dataset, "Train", res_train, epoch)

                res_valid = test(model, valid_loader)
                save_result_DTI(dataset, "Valid", res_valid, epoch)

                res_test = test(model, test_loader)
                all_res_test.append(res_test)
                save_result_DTI(dataset, "Test", res_test, epoch)

                # 提取前六列数据
                columns_to_save = ['Acc', 'Pre', 'Recall', 'AUC', 'AUPR', 'LOSS']
                data_to_save = res_test[:6]
                # -----------------------------------------------------------------------------
                # 创建 DataFrame
                df_to_save = pd.DataFrame([data_to_save], columns=columns_to_save)
                # 创建每个折的结果保存路径
                fold_result_path = f'./实验结果/Davis/添加药物图像和文本特征和靶标的二级结构信息/Fold_{fold_i}/'
                os.makedirs(fold_result_path, exist_ok=True)

                # 写入测试结果到CSV文件
                csv_path = fold_result_path + f'Davis_result_std_fold_{fold_i}.csv'
                # df_to_save.to_csv(csv_path, index=False, mode='a', header=True)
                # 写入列名（第一次迭代）或追加数据（后续迭代）
                if is_first_iteration:
                    df_to_save.to_csv(csv_path, index=False, mode='a', header=True)
                    is_first_iteration = False
                else:
                    df_to_save.to_csv(csv_path, index=False, mode='a', header=False)

                # 检查当前 AUC 是否是最大值
                if res_test[3] > best_auc:
                    best_auc = res_test[3]
                    best_auc_fpr, best_auc_tpr = res_test[6], res_test[7]
                    # 保存到 auc_fpr_tpr.csv 文件
                    df_auc_fpr_tpr = pd.DataFrame({'fpr': best_auc_fpr, 'tpr': best_auc_tpr})

                    auc_fpr_tpr_path = fold_result_path + f'Davis_auc_fpr_tpr_std_fold_{fold_i}.csv'
                    df_auc_fpr_tpr.to_csv(auc_fpr_tpr_path, index=False)

                # 检查当前 AUPR 是否是最大值
                if res_test[4] > best_aupr:
                    best_aupr = res_test[4]
                    best_aupr_recall, best_aupr_precision = res_test[8], res_test[9]
                    # 保存到 aupr_precision_recall.csv 文件
                    df_aupr_precision_recall = pd.DataFrame({'recall': best_aupr_recall, 'precision': best_aupr_precision})
                    aupr_recall_precision_path = fold_result_path + f'Davis_aupr_recall_precision_std_fold_{fold_i}.csv'
                    df_aupr_precision_recall.to_csv(aupr_recall_precision_path, index=False)

                if res_valid[3] > valid_max_AUC:
                    valid_max_AUC = res_valid[3]
                    decline = 0
                    save_path = './save_model/{}/{}_valid_best_checkpoint_{}.pth'.format(dataset, args.substructure,
                                                                                         epoch)
                    torch.save(model.state_dict(), save_path)
                else:
                    decline = decline + 1
                    if decline >= args.count_decline:
                        print("EarlyStopping !!!")
                        break

                #-----------------------------------------------------------------------------
                if epoch % args.lr_step_size == 0:  # 每隔n次学习率衰减一次
                    optimizer.param_groups[0]['lr'] *= args.lr_gamma

            for res in all_res_test:
                Acc = res[0]
                Acc_List_stable.append(Acc)

                Pre = res[1]
                Pre_List_stable.append(Pre)

                Recall = res[2]
                Recall_List_stable.append(Recall)

                AUC = res[3]
                AUC_List_stable.append(AUC)

                AUPR = res[4]
                AUPR_List_stable.append(AUPR)


        show_result(Acc_List_stable, Pre_List_stable, Recall_List_stable, AUC_List_stable, AUPR_List_stable)


'''