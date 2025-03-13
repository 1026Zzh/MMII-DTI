from tqdm import tqdm
from collections import defaultdict
import json
import pickle
import os
import numpy as np

target_dict = defaultdict(lambda: len(target_dict))

#这段函数主要是提取出靶标的氨基酸序列
def read_target_interaction(path, flag):
    all_target = []
    with open(path, 'r') as f:
        data = f.read().strip().split('\n')
        f.close()
    for item in data:
        target = str(item).split(' ')[flag]
        if target not in all_target:
            all_target.append(target)
    return all_target


def read_target_affinity(path1, path2, flag):
    all_target = []
    with open(path1, 'r') as f:
        data1 = f.read().strip().split('\n')
        f.close()
    with open(path2, 'r') as f:
        data2 = f.read().strip().split('\n')
        f.close()
    data = data1 + data2
    for item in data:
        target = str(item).split(',')[flag]
        if target not in all_target:
            all_target.append(target)
    return all_target

#该函数用于将目标序列（通常是蛋白质的氨基酸序列）拆分成 n-gram 组，并将每个 n-gram 组转换为数字编码。
def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [target_dict[sequence[i:i + ngram]] + 1
             for i in range(len(sequence) - ngram + 1)]
    return words

# 靶标二级结构的编码
SSESET = {"e": 1, "c": 2, "t": 3, "h": 4, "b":5, "s":6, "g":7, "i":8}
SSELEN = 8
max_seq_len = 1000
# 用于对蛋白质二级结构编码进行one-hot编码  smi_ch_ind 表示 SSESET这个字典
def one_hot_sse(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)), dtype="int8")
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch]) - 1] = 1

    return X  # .tolist()

if __name__ == '__main__':
    # datasets = ['human']
    # datasets = ['Davis']
    # datasets = ['celegans']
    # datasets = ['KIBA']
    datasets = ['Davis_new']
    # datasets = ['KIBA_new']
    all_target = []
    all_target_second_structure = []

    # ---------------------Step 1: extracting target sequences--------------------------------
    for dataset in tqdm(datasets):
        flag = 3
        # if dataset in ['human']:
        # if dataset in ['Davis']:
        # if dataset in ['celegans']:
        # if dataset in ['KIBA']:
        if dataset in ['Davis_new']:
        # if dataset in ['KIBA_new']:
            filename = "./RawData/interaction/case_study_target_Davis/case_study_target_Davis_test/{}.txt".format(dataset)
            # filename = "./RawData/interaction/{}.txt".format(dataset)
            # if dataset in ['Davis', 'KIBA']:
            #     flag = 3
            targets = read_target_interaction(filename, flag)
            all_target += targets
        else:
            filename_train = "./RawData/affinity/{}/train.txt".format(dataset)
            filename_test = "./RawData/affinity/{}/test.txt".format(dataset)
            targets = read_target_affinity(filename_train, filename_test, flag)
            all_target += targets
    all_target = list(set(all_target))

    with open('./common/case_study_target_Davis_test/all_target.txt', 'w') as f:
        for target in all_target:
            f.write(str(target) + '\n')
        f.close()
    print("Successfully extracting target... ")

    #-----------------------Step 2: 抽取出靶标的二级结构信息序列-----------------------------------
    for dataset in tqdm(datasets):
        flag = 4
        # if dataset in ['human']:
        # if dataset in ['Davis']:
        # if dataset in ['celegans']:
        # if dataset in ['KIBA']:
        if dataset in ['Davis_new']:
        # if dataset in ['KIBA_new']:
            filename = "./RawData/interaction/case_study_target_Davis/case_study_target_Davis_test/{}.txt".format(dataset)
            # filename = "./RawData/interaction/{}.txt".format(dataset)
            # if dataset in ['Davis', 'KIBA']:
            #     flag = 4
            targets_second_structure = read_target_interaction(filename, flag)
            all_target_second_structure += targets_second_structure
        else:
            filename_train = "./RawData/affinity/{}/train.txt".format(dataset)
            filename_test = "./RawData/affinity/{}/test.txt".format(dataset)
            targets = read_target_affinity(filename_train, filename_test, flag)
            all_target_second_structure += targets_second_structure
    all_target_second_structure = list(set(all_target_second_structure))
    print("Successfully extracting target second structure... ")


    # ----------------------Step 3: encoding target sequences------------------------------
    all_residues = []
    for target in tqdm(all_target):
        residues = split_sequence(target, 3)
        all_residues.append(residues)
    with open('./common/case_study_target_Davis_test/residues_id.txt', 'w') as f:   #这个文件应该代表靶标按3个长度进行分解，得到的子序列，将子序列按序号进行的索引。
        json.dump(all_residues, f)
        f.close()

    os.makedirs('./common/case_study_target_Davis_test/dict', exist_ok=True)

    with open('./common/case_study_target_Davis_test/dict/target_dict', 'wb') as f:       # 保存子图字典
        pickle.dump(dict(target_dict), f)
        f.close()

    print("Successfully encoding target sequence... ")

    # ----------------------Step 4: 对靶标二级结构信息进行one-hot编码------------------------------
    all_tss = []
    for target_second_structures in tqdm(all_target_second_structure):
        # print("target_second_structures:",target_second_structures)
        # print("target_second_structures_len:", len(target_second_structures))# 表示靶标二级结构序列的长度
        # all_tss.append(one_hot_sse(target_second_structures, max_seq_len, SSESET).tolist())
        X = one_hot_sse(target_second_structures, max_seq_len, SSESET).tolist()
        # print("X:",X)# 表示每个靶标的二级结构序列通过one-hot编码后的二维矩阵形式
        # print("X_len:", len(X))# 固定长度为1000
        all_tss.append(X)# 所有靶标二级结构序列的二维矩阵合并成一个三维矩阵形式
        # print("all_tss:", all_tss)
        # print("all_tss:",len(all_tss)) #长度是靶标的数量 KIBA：225
    # 将所有靶标的三维矩阵转换为二维矩阵
    all_tss = np.vstack(all_tss).tolist()
    # print("all_tss:", all_tss)
    # print("all_tss:", len(all_tss))# KIBA：225*1000=225000
    with open('./common/case_study_target_Davis_test/tss_id.txt', 'w') as f:
        json.dump(all_tss, f)
        f.close()
    print("Successfully encoding target second structure... ")

    # os.makedirs('./common/Davis/dict', exist_ok=True)
    #
    # with open('./common/Davis/dict/target_dict', 'wb') as f:  # 保存子图字典
    #     pickle.dump(dict(target_dict), f)
    #     f.close()
    #
    # print("Successfully encoding target sequence... ")
