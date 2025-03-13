import numpy as np
from torch.utils.data import Dataset
from utils import get_train_data, get_positions, get_all_adjs, to_float_tensor, to_long_tensor, positions_match, batch_pad
from torch.nn.utils.rnn import pad_sequence
import torch
from torchvision import transforms
from PIL import Image
import os

class MyDataset(Dataset): #继承自 torch.utils.data.Dataset
    def __init__(self, common_data, train_data, args):
        self.args = args
        self.all_smiles_idx = common_data[0]
        self.all_smiles_new = common_data[1]
        self.all_targets_idx = common_data[2]
        self.all_residues = common_data[3]
        self.all_substructures = common_data[4]
        self.all_skeletons = common_data[5]
        self.all_mols = common_data[6]
        self.all_atoms_idx = common_data[7]
        self.all_marks = common_data[8]
        self.all_gram = common_data[9] #药物文本特征路径
        self.all_image = common_data[10] #药物图像路径
        self.all_target_second_structures = common_data[11] #靶标的二级结构信息

        self.transformImg = transforms.Compose([
            transforms.ToTensor()])

        self.all_pos_idx, self.all_positions = get_positions(self.all_mols)
        self.all_adjs = get_all_adjs(self.all_mols)



        self.train_drugs = train_data[0]
        self.train_targets = train_data[1]
        self.train_labels = train_data[2]

        # target data (氨基酸序列)
        self.train_residues = get_train_data(self.all_residues, self.all_targets_idx, self.train_targets)

        # target data (靶标二级结构序列)
        self.train_target_second_structures = get_train_data(self.all_target_second_structures, self.all_targets_idx, self.train_targets)

        # drug data
        #药物图像的添加
        img_paths = self.all_image
        drugimage = torch.stack([self.transformImg(Image.open(img_path).convert('RGB')) for img_path in img_paths])
        self.train_image = get_train_data(drugimage, self.all_smiles_idx, self.train_drugs)

        #药物文本特征的添加
        self.train_gram = get_train_data(self.all_gram, self.all_smiles_idx, self.train_drugs)

        self.train_substructures = get_train_data(self.all_substructures, self.all_smiles_idx, self.train_drugs)
        self.train_skeletons = get_train_data(self.all_skeletons, self.all_smiles_idx, self.train_drugs)
        self.train_atoms_idx = get_train_data(self.all_atoms_idx, self.all_smiles_idx, self.train_drugs)
        self.train_marks = get_train_data(self.all_marks, self.all_smiles_idx, self.train_drugs)
        self.train_pos_idx = get_train_data(self.all_pos_idx, self.all_smiles_idx, self.train_drugs)
        self.train_positions = get_train_data(self.all_positions, self.all_smiles_idx, self.train_drugs)
        self.train_adjs = get_train_data(self.all_adjs, self.all_smiles_idx, self.train_drugs)

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, item):
        residues_f = np.array(self.train_residues[item][:self.args.max_target])
        substructures_f = np.array(self.train_substructures[item])
        skeletons_f = np.array(self.train_skeletons[item])
        atoms_idx = self.train_atoms_idx[item]
        marks = self.train_marks[item]
        pos_idx = self.train_pos_idx[item]
        positions = self.train_positions[item]
        adjs = self.train_adjs[item]
        label = self.train_labels[item]
        druggram = self.train_gram[item]

        drugimage = self.train_image[item] #tensor
        # print("drugimage:", drugimage.shape) #drugimage: torch.Size([3, 256, 256])
        target_second_structures_f = np.array(self.train_target_second_structures[item][:self.args.max_target]) # 新增的靶标的二级结构序列


        return residues_f, substructures_f, skeletons_f, atoms_idx, marks, pos_idx, positions, adjs, label, druggram, drugimage, target_second_structures_f



def collate_fn(batch, device):
    # 两个字典，用于分别存储药物和靶蛋白的数据
    drug, target = {}, {}

    #存储了每个样本的蛋白质残基数据
    batch_residues = []
    batch_substructures, batch_skeletons, batch_atoms_idx, batch_marks = [], [], [], []
    batch_positions, batch_adjs, len_adjs = [], [], []
    batch_labels = []
    batch_druggram = []
    batch_images = []
    batch_target_second_structures = []

    len_residues, len_substructures, len_skeletons = [], [], []
    len_druggram = []
    len_images = []
    len_target_second_structures = []

    for item in batch:
        batch_residues.append(to_long_tensor(item[0]))
        len_residues.append(len(item[0]))
        batch_substructures.append(to_long_tensor(item[1]))
        len_substructures.append(len(item[1]))
        batch_skeletons.append(to_long_tensor(item[2]))
        len_skeletons.append(len(item[2]))
        batch_atoms_idx.append(item[3])
        batch_marks.append(to_long_tensor(item[4]))
        positions_new = positions_match(item[3], item[4], item[5], item[6])
        batch_positions.append(to_float_tensor(positions_new))
        len_adjs.append(len(item[7]))
        batch_adjs.append(item[7])

        batch_labels.append(item[8])
        # batch_druggram.append((item[9]))  # 将药物文本特征信息添加到列表
        batch_druggram.append(to_long_tensor(item[9]))  # 将药物文本特征信息添加到列表
        len_druggram.append(len(item[9]))
        batch_images.append((item[10]))  # 将药物图像信息添加到列表
        # batch_images.append((item[10]))  # 将药物图像信息添加到列表
        # len_images.append(len(item[10]))
        batch_target_second_structures.append(to_long_tensor(item[11]))
        len_target_second_structures.append(len(item[11]))

    #找到了样本中最短的药物三维原子结构数据，并记录下其索引 min_skeleton_idx。
    min_skeleton_idx = len_skeletons.index(min(len_skeletons))

    #该批次将存储了经过填充的药物三维原子结构的邻接矩阵。
    batch_adjs = to_long_tensor(batch_pad(batch_adjs, max(len_adjs))).to(device)

    batch_residues = pad_sequence(batch_residues, batch_first=True, padding_value=0).to(device)
    batch_substructures = pad_sequence(batch_substructures, batch_first=True, padding_value=0).to(device)
    batch_skeletons = pad_sequence(batch_skeletons, batch_first=True, padding_value=0).to(device)
    batch_positions = pad_sequence(batch_positions).permute(1, 0, 2).to(device)
    batch_druggram = pad_sequence(batch_druggram, batch_first=True, padding_value=0).to(device)
    # batch_images = pad_sequence(batch_images, batch_first=True, padding_value=0).to(device)
    batch_target_second_structures = pad_sequence(batch_target_second_structures, batch_first=True, padding_value=0).to(device)


    batch_residues_masks = to_long_tensor(np.zeros((batch_residues.shape[0], batch_residues.shape[1]))).to(device)
    batch_substructures_masks = to_long_tensor(np.zeros((batch_substructures.shape[0], batch_substructures.shape[1]))).to(device)
    for i in range(len(len_residues)):
        batch_residues_masks[i, :len_residues[i]] = 1
        batch_substructures_masks[i, :len_substructures[i]] = 1

    # 新增的靶标二级结构信息
    batch_target_second_structures_masks = to_long_tensor(np.zeros((batch_target_second_structures.shape[0], batch_target_second_structures.shape[1]))).to(device)
    # batch_substructures_masks = to_long_tensor(
    #     np.zeros((batch_substructures.shape[0], batch_substructures.shape[1]))).to(device)
    for i in range(len(len_target_second_structures)):
        batch_target_second_structures_masks[i, :len_target_second_structures[i]] = 1
        # batch_substructures_masks[i, :len_substructures[i]] = 1

    # 添加的药物文本特征
    batch_drug_features_mask = to_long_tensor(np.zeros((batch_druggram.shape[0], batch_druggram.shape[1]))).to(device)
    for i in range(len(len_residues)):
        batch_drug_features_mask[i, :len_residues[i]] = 1
        batch_drug_features_mask[i, :len_druggram[i]] = 1


    target['residues'] = batch_residues
    target['residues_masks'] = batch_residues_masks
    drug['substructures'] = batch_substructures
    drug['substructures_masks'] = batch_substructures_masks
    drug['skeletons'] = batch_skeletons
    drug['marks'] = batch_marks
    drug['positions'] = batch_positions
    drug['adjs'] = batch_adjs
    drug['padding_tensor_idx'] = min_skeleton_idx
    drug['drug_features'] = batch_druggram  # 添加药物的文本特征信息
    drug['drug_features_mask'] = batch_drug_features_mask # 添加药物的文本特征掩码
    drug['drug_images'] = batch_images  # 添加药物的图像信息
    # drug['drug_images_mask'] = batch_drug_images_mask  # 添加药物的图像特征掩码
    target['tss'] = batch_target_second_structures # 靶标二级结构特征
    target['tss_masks'] = batch_target_second_structures_masks # 靶标二级结构特征掩码

    return target, drug, batch_labels



