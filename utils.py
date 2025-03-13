import json
import sys
from rdkit import Chem
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from torchvision import transforms


def read_file(path):
    with open(path, 'r') as f:
        data = f.read().strip().split('\n')
    f.close()
    return data


def read_file_by_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    f.close()
    return data


def load_common_data(subs):
    print("Loading common dataset (about 2 minutes)...")
    # load drug and targets
    # all_smiles = read_file('./common/Davis/all_smiles.txt')
    all_smiles = read_file('./common/KIBA/all_smiles.txt')

    # all_smiles_new = read_file('./common/Davis/all_smiles_new.txt')
    all_smiles_new = read_file('./common/KIBA/all_smiles_new.txt')


    # all_targets = read_file('./common/Davis/all_target.txt')
    all_targets = read_file('./common/KIBA/all_target.txt')

    # The features of the target.
    # residues = read_file_by_json('./common/Davis/residues_id.txt')  #靶标的氨基酸序列进行分解时的索引
    residues = read_file_by_json('./common/KIBA/residues_id.txt')

    # 新增加的靶标的二级结构信息
    # tss = read_file_by_json('./common/Davis/tss_id.txt')  # 靶标的氨基酸序列进行分解时的索引
    tss = read_file_by_json('./common/KIBA/tss_id.txt')

    # The features of the drug 2D substructure.
    subs = subs.upper()
    if subs == 'ICMF':
        # substructures = read_file_by_json('./common/Davis/substructure_icmf_id.txt')
        substructures = read_file_by_json('./common/KIBA/substructure_icmf_id.txt')
    elif subs == 'ESPF':
        # substructures = read_file_by_json('./common/Davis/substructure_espf_id.txt')
        substructures = read_file_by_json('./common/KIBA/substructure_espf_id.txt')
    else:
        print('The parameter of drug 2D substructure is wrong!!! (Please choose for ICMF and ESFP)')
        sys.exit(1)

    # 加入药物图像信息
    # drug_img_path = "./common/Davis/image/Img_256_256/img_inf_data"
    drug_img_path = "./common/KIBA/image/Img_256_256/img_inf_data"

    # 得到药物图像路径
    drug_image = get_img_path(drug_img_path)

    # 加入药物文本信息
    # druggram = read_file_by_json('./common/Davis/Davis_smiles_1_gram.txt')
    druggram = read_file_by_json('./common/KIBA/KIBA_smiles_1_gram.txt')


    # The features of the drug 3D.
    # skeletons = read_file_by_json('./common/Davis/skeletons_id.txt') #药物的骨架
    skeletons = read_file_by_json('./common/KIBA/skeletons_id.txt')


    # mols = list(Chem.SDMolSupplier('./common/Davis/all_mols.sdf')) #SMILES表示的分子转化为具有三维坐标的分子
    mols = list(Chem.SDMolSupplier('./common/KIBA/all_mols.sdf'))

    # atoms_idx = read_file_by_json('./common/Davis/atoms_idx.txt') #药物中原子的索引
    atoms_idx = read_file_by_json('./common/KIBA/atoms_idx.txt')


    # marks = read_file_by_json('./common/Davis/marks.txt') #氢原子和非氢原子位置；氢原子代表1，非氢原子代表0
    marks = read_file_by_json('./common/KIBA/marks.txt')
    return [all_smiles, all_smiles_new, all_targets, residues, substructures, skeletons, mols, atoms_idx, marks, druggram, drug_image, tss]


def shuffle_dataset(data):
    np.random.seed(1234)
    np.random.shuffle(data)
    return data



def load_train_data_DTI(dataset):
    drugs, targets, labels = [], [], []
    fpath = './RawData/interaction/{}.txt'.format(dataset)
    train_data = shuffle_dataset(read_file(fpath))
    print("Loading train dataset...")
    if dataset == 'human' or dataset == 'celegans':
    # if dataset == 'human':
        for item in tqdm(train_data):
            data = str(item).split(' ')
            drugs.append(data[0])
            targets.append(data[1])
            labels.append(int(data[2]))
    elif dataset == 'Davis' or dataset == 'KIBA':
    # elif dataset == 'Davis':
        for item in tqdm(train_data):
            data = str(item).split(' ')
            drugs.append(data[2])
            targets.append(data[3])
            labels.append(int(data[4]))
    return [drugs, targets, labels]


def load_train_data_CPA(dataset):
    drugs_train, targets_train, labels_train = [], [], []
    drugs_test, targets_test, labels_test = [], [], []
    fpath_train = './RawData/affinity/{}/train.txt'.format(dataset)
    fpath_test = './RawData/affinity/{}/test.txt'.format(dataset)
    train_data = shuffle_dataset(read_file(fpath_train))
    test_data = shuffle_dataset(read_file(fpath_test))

    print("Loading train dataset...")
    for item in tqdm(train_data):
        data = str(item).split(',')
        drugs_train.append(data[0])
        targets_train.append(data[1])
        labels_train.append(float(data[2]))

    print("Loading test dataset...")
    for item in tqdm(test_data):
        data = str(item).split(',')
        drugs_test.append(data[0])
        targets_test.append(data[1])
        labels_test.append(float(data[2]))
    return [drugs_train, targets_train, labels_train], [drugs_test, targets_test, labels_test]


def get_train_data(all_data, all_data_ids, train_data_ids):
    all_data_dict = {value: idx for idx, value in enumerate(all_data_ids)}

    train_data = []
    for v in train_data_ids:
        idx = all_data_dict[v]
        train_data.append(all_data[idx])
    return train_data


def get_positions(mols):
    all_pos_idx = []
    all_positions = []
    print("Loading positions...")
    for mol in tqdm(mols):
        conformer = mol.GetConformer() #从当前分子对象 mol 中获取其构象（三维坐标）信息

        # extract atomic space coordinates
        atoms = mol.GetAtoms() #获取分子对象中包含的所有原子
        atoms_idx = [atom.GetIdx() for atom in atoms] #每个原子的索引
        positions = [conformer.GetAtomPosition(idx) for idx in atoms_idx] #通过使用原子索引列表 atoms_idx.txt 和 all_mol.sdf 对象，可以获得每个原子的坐标信息
        all_pos_idx.append(atoms_idx)
        all_positions.append(positions)
    return all_pos_idx, all_positions #分别包含了每个原子的位置索引和每个原子对应的三维空间坐标。


def get_edge_index(mols):
    edge_index = [] #用于存储原子之间的连接关系
    print("Loading edge index...")
    for mol in tqdm(mols): #从all_mol.sdf文件中进行遍历
        bonds = mol.GetBonds() #获取当前 mol 中的所有化学键（连接）

        e1, e2 = [], []
        for bond in bonds:
            e1.append(bond.GetBeginAtomIdx())
            e1.append(bond.GetEndAtomIdx())
            e2.append(bond.GetEndAtomIdx())
            e2.append(bond.GetBeginAtomIdx())
        edge_index.append([e1,e2])
    return edge_index #获取药物分子中 每个原子 之间的连接关系（边缘索引）


def get_all_adjs(mols):
    adjs = [] #用于存储药物分子的邻接矩阵
    print("Loading adjacency...")
    for mol in tqdm(mols):
        mol = Chem.AddHs(mol) #将当前分子 mol 添加氢原子，以确保分子中包含氢原子
        '''
        计算当前 mol 的邻接矩阵，并将其添加到 adjs 列表中。
        邻接矩阵表示了分子中原子之间的连接关系，通常以 0 和 1 表示连接或非连接。
        如果两个原子之间存在化学键，邻接矩阵中的相应元素为 1；否则，为 0。
        '''
        adjs.append(Chem.GetAdjacencyMatrix(mol)) #用于生成分子的邻接矩阵，表示分子中原子之间的连接关系。
    return adjs


def batch_pad(datas, N):
    data = np.zeros((len(datas), N, N))
    for i, a in enumerate(datas):
        n = a.shape[0]
        data[i, :n, :n] = a
    return data


def to_float_tensor(data):
    return torch.FloatTensor(data)


def to_long_tensor(data):
    return torch.LongTensor(data)



def positions_match(atoms_ix, marks, pos_idx, positions):
    pos_new = []
    for i, idx in enumerate(atoms_ix):
        if marks[i] == 1:
            pos_new.append(positions[pos_idx.index(idx)])
    return pos_new


def get_five_fold_idx(lens, fold_i, K_FOLD):
    idx = [id for id in range(lens)]

    fold_size = lens // K_FOLD
    val_start = fold_i * fold_size
    val_end = (fold_i + 1) * fold_size

    train_idx = idx[:val_start] + idx[val_end:]
    test_idx = idx[val_start:val_end]

    return train_idx, test_idx


def get_test_idx(lens, ratio, batchs):
    idx = np.arange(lens)
    num = int(lens/batchs * (1-ratio))
    num_train = num * batchs
    idx_train, idx_test = idx[:num_train], idx[num_train:]
    return idx_train, idx_test

def get_img_path(img_path):
    imgs = []
    with open(img_path, "r") as f:
        lines = f.read().strip().split("\n")
        for line in lines:
            imgs.append(line.split("\t")[0])
    return imgs #输出结果是[‘图像路径0001’, ‘图像路径2’ ...,‘图像路径2726’]




