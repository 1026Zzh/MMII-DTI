from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

'''
这段代码主要用于将SMILES表示的分子转化为具有三维坐标的分子，
并将其保存在SDF文件中，以便进行后续的化学分析和建模。
'''

if __name__ == '__main__':
    all_smiles_new = []
    all_mols = []

    # Step 1:
    with open('./common/case_study_target_Davis_test/all_smiles.txt', "r") as f:
        all_smiles = f.read().strip().split('\n')
    f.close()

    with open('./common/case_study_target_Davis_test/all_smiles_new.txt', 'w') as f:
        for smiles in tqdm(all_smiles):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
            all_smiles_new.append(smiles)
            f.write(str(smiles) + '\n')
        f.close()
    print("Successfully extracting SMILES... ")

    # Step 2:
    for smiles in tqdm(all_smiles_new):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=1234)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except ValueError:
            AllChem.EmbedMultipleConfs(mol, numConfs=3)
            pass
        all_mols.append(Chem.RemoveHs(mol))

    w1 = Chem.SDWriter('./common/case_study_target_Davis_test/all_mols.sdf')  #这个all_mols.sdf文件代表了药物中各个原子之间的三维坐标信息以及原子之间的连接关系
    for m in all_mols:
        w1.write(m)

    print("Successfully extracting Coordinates... ")
