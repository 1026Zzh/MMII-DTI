# MMII_DTI
Drug-target interaction prediction based on multimodal information integration

Functions of each script:
00_extract_all_smiles.py: normalized drug SMILES sequences.
01_create_smiles_and_sdf.py: it is mainly used to convert molecules represented by SMILES into molecules with three-dimensional coordinates and save them in SDF files for subsequent chemical analysis and modeling.
02_encode_drug_2D.py: the main function is to decompose the ICMF of the drug substructure, and to construct a dictionary of two decomposition methods, in the form of {substructure: index value}, and to obtain a representation of the index value of the substructure after the drug decomposition by two methods.
03_encode_drug_3D.py: the three-dimensional structure that encodes the drug.
04_1_encode_target.py: the amino acid sequence of the target was processed.
04_2_encode_target_add_target_secondary_structure.py: the secondary structure of the target is processed.
05_smile_to_image.py: first, the SMILES format of drugs was screened, then the SMILES chemical structure strings were converted into chemical molecular images, and these image files were saved.
06_1_smile_to_features.py: the SMILES chemical structure string is converted into the characteristic information of the molecule, and this characteristic information is saved to the file.
06_2_smile_k_gram.py: the SMILES sequences of drugs were mapped.
07_2_train_Davis.py: the main function.
Davis: Davis dataset

Scholars, if there are shortcomings, welcome to discuss. Email: zonghaoo@163.com
