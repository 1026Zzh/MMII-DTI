from tqdm import tqdm
from collections import defaultdict
import json
import pickle
import os

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


if __name__ == '__main__':
    # datasets = ['human']
    datasets = ['Davis']
    # datasets = ['celegans']
    # datasets = ['KIBA']
    all_target = []

    # Step 1: extracting target sequences
    for dataset in tqdm(datasets):
        flag = 3
        # if dataset in ['human']:
        if dataset in ['Davis']:
        # if dataset in ['celegans']:
        # if dataset in ['KIBA']:
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

    # Step 2: encoding target sequences
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
