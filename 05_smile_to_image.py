from PIL import Image
import os
from rdkit.Chem import Draw
from rdkit import Chem

'''
该脚本这段代码的功能是：
    1、先对药物的SMILES格式进行数据筛选
    2、将SMILES化学结构字符串转换为化学分子图像，并保存这些图像文件
'''

#该函数从数据文件中读取SMILES字符串，将每个SMILES字符串转换为相应的化学分子图像，并将图像保存到指定目录
def smile2pic(file_path, file_data):
    with open(file_data, "r") as f:
        data_list = f.read().strip().split("\n")

    """Exclude data contains '.' in the SMILES format."""  # # The '.' represents multiple chemical molecules
    # data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

    smiles = []
    for i, data in enumerate(data_list):
        if i % 100 == 0:
            print('/'.join(map(str, [i + 1, len(data_list)])))

        smile = data.strip().split(" ")[0]

        mol = Chem.MolFromSmiles(smile)
        # 规范化SMILES表达式
        canonical_smi = Chem.MolToSmiles(mol)
        canonical_mol = Chem.MolFromSmiles(canonical_smi)
        # 生成分子图像
        img = Draw.MolToImage(mol, size=(pic_size, pic_size), wedgeBonds=False)
        number = str(i + 1)
        number = number.zfill(len(str(len(data_list))))

        smiles += smile

        save_name = file_path + "/" + number + ".png"
        img.save(save_name)


#该函数用于统计指定目录下的图像文件数量，并将图像文件名和路径信息写入一个文本文件中
def pic_info(file_path):
    file_list = os.listdir(file_path)
    num = 0
    for pic in file_list:
        if ".png" in pic:
            num += 1
    str_len = len(str(num))
    print(str_len)
    print(file_path)
    with open(file_path + "/img_inf_data", "w") as f:
        for i in range(num):
            number = str(i + 1)
            number = number.zfill(len(str(len(file_list))))
            if i == num - 1:
                f.write(file_path + "/" + number + ".png" + "\t" + number + ".png")
            else:
                f.write(file_path + "/" + number + '.png' + "\t" + number + '.png' + "\n")


if __name__ == '__main__':
    dataset_name = "all_smiles_new"
    #图像的尺寸：256*256
    pic_size = 256
    # data_root = "./common/Human/" + dataset_name
    # data_root = "./common/Davis/" + dataset_name
    # data_root = "./common/celegans/" + dataset_name
    # data_root = "./common/KIBA/" + dataset_name
    data_root = "./common/case_study_target_Davis_test/" + dataset_name
    train_file = data_root + ".txt"


    # train_path = "./common/Human/image/"
    # train_path = "./common/Davis/image/"
    # train_path = "./common/celegans/image/"
    # train_path = "./common/KIBA/image/"
    train_path = "./common/case_study_target_Davis_test/image/"
    if not os.path.exists(train_path):
        os.makedirs(train_path)


    pic_train_path = train_path + "Img_" + str(pic_size) + "_" + str(pic_size)
    if not os.path.exists(pic_train_path):
        os.makedirs(pic_train_path)

    smile2pic(pic_train_path, train_file)
    print("Train_Pic generated.size=", pic_size, "*", pic_size, "----")

    pic_info(pic_train_path)

