import os
from rdkit import RDLogger
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

# 去重复功能
def read_smiles_interaction(path, flag):
    all_smiles = []
    with open(path, 'r') as f:
        data = f.read().strip().split('\n')
        f.close()
    for item in data:
        smiles = str(item).split(' ')[flag]
        if smiles not in all_smiles:
            all_smiles.append(smiles)
    return all_smiles


def read_smiles_affinity(path1, path2, flag=0):
    all_smiles = []
    with open(path1, 'r') as f:
        data1 = f.read().strip().split('\n')
        f.close()
    with open(path2, 'r') as f:
        data2 = f.read().strip().split('\n')
        f.close()
    data = data1 + data2
    for item in data:
        smiles = str(item).split(',')[flag]
        if smiles not in all_smiles:
            all_smiles.append(smiles)
    return all_smiles


if __name__ == '__main__':
    # datasets = ['human']
    datasets = ['Davis']
    # datasets = ['celegans']
    # datasets = ['KIBA']
    all_smiles = []

    # Step 1:
    for dataset in tqdm(datasets):
        flag = 2
        # if dataset in ['human']:
        if dataset in ['Davis']:
        # if dataset in ['celegans']:
        # if dataset in ['KIBA']:
            filename = "./RawData/interaction/case_study_target_Davis/case_study_target_Davis_test/{}.txt".format(dataset)
            # filename = "./RawData/interaction/{}.txt".format(dataset)
            # if dataset in ['Davis', 'KIBA']:
            #     flag = 2
            dataset_smiles = read_smiles_interaction(filename, flag)
            all_smiles += dataset_smiles
        else:
            filename_train = "./RawData/affinity/{}/train.txt".format(dataset)
            filename_test = "./RawData/affinity/{}/test.txt".format(dataset)
            dataset_smiles = read_smiles_affinity(filename_train, filename_test, flag)
            all_smiles += dataset_smiles
    all_smiles = list(set(all_smiles))

    os.makedirs('./common', exist_ok=True)

    with open('./common/case_study_target_Davis_test/all_smiles.txt', 'w') as f:
        for smiles in all_smiles:
            f.write(str(smiles) + '\n')
        f.close()
    print("Successfully extracting primitive SMILES... ")
