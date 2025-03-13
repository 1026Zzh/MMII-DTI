from PIL import Image
import os
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from rdkit import Chem


'''
该脚本的功能是将SMILES化学结构字符串转换为分子的特征信息，并将这些特征信息保存到文件中
'''

def smile2feature(data_root, type, file_data):
    with open(file_data, "r") as f:
        data_list = f.read().strip().split("\n")

    """Exclude data contains '.' in the SMILES format."""  # The '.' represents multiple chemical molecules
    # data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

    smile_features = []
    file_name = data_root + "/" + type + "_smile_features.txt"
    with open(file_name, "w") as w:
        for i, data in enumerate(data_list):
            if i % 50 == 0:
                print('/'.join(map(str, [i + 1, len(data_list)])))

            fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
            smile = data.strip().split(" ")[0]
            mol = Chem.MolFromSmiles(smile)
            feats = factory.GetFeaturesForMol(mol)

            line = ""
            for f in feats:
                s = str(f.GetFamily())
                s += " " + str(f.GetType())
                s += " " + str(f.GetAtomIds())
                # s += " " + str(f.GetId())
                line += s + " "
            line += "\n"
            w.write(line)


if __name__ == '__main__':
    dataset_name = "all_smiles_new"
    data_root = "./common/case_study_target_Davis_test/"
    train_file = data_root + dataset_name + ".txt"


    # smile2feature(data_root, "human", train_file)
    smile2feature(data_root, "Davis", train_file)

