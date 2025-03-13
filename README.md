# MMII_DTI
Drug-target interaction prediction based on multimodal information integration

各个脚本的功能：
00_extract_all_smiles.py：规范化药物SMILES序列。
01_create_smiles_and_sdf.py：主要用于将SMILES表示的分子转化为具有三维坐标的分子，并将其保存在SDF文件中，以便进行后续的化学分析和建模。
02_encode_drug_2D.py：主要功能是对药物亚结构进行ICMF的分解，以及构建两个分解方法的字典，形式：{子结构：索引值}，还有得到两种方法进行药物分解后子结构的索引值的表示。
03_encode_drug_3D.py：编码药物的三维结构。
04_1_encode_target.py：对靶标的氨基酸序列进行处理。
04_2_encode_target_add_target_secondary_structure.py：对靶标的二级结构进行处理。
05_smile_to_image.py：先对药物的SMILES格式进行数据筛选，再将SMILES化学结构字符串转换为化学分子图像，并保存这些图像文件。
06_1_smile_to_features.py：将SMILES化学结构字符串转换为分子的特征信息，并将这些特征信息保存到文件中。
06_2_smile_k_gram.py：对药物的SMILES序列进行映射。
07_2_train_Davis.py：main函数。

各位学者，如有不足之处，欢迎讨论。邮箱：zonghaoo@163.com
