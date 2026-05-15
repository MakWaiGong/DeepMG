import pandas as pd
import pickle
from rdkit import Chem
# 路径配置
smiles2rep_path = "smiles2rep.pkl"
dataset7_path = "dataset7.xlsx"

# 加载 smiles2rep.pkl
with open(smiles2rep_path, "rb") as f:
    smiles2rep = pickle.load(f)

# 加载 dataset7
df = pd.read_excel(dataset7_path)

# 假设SMILES列名为 "SMILES"，根据需要替换
smiles_list = df["Smiles"].tolist()

# 对比 smiles 是否在 smiles2rep 中
not_in_smiles2rep = []
for smi in smiles_list:
    if smi not in smiles2rep:
        not_in_smiles2rep.append(smi)

print(f"共{len(smiles_list)}个SMILES，其中{len(not_in_smiles2rep)}个不在smiles2rep中。")
if not_in_smiles2rep:
    print("以下SMILES不在smiles2rep中：")
    for smi in not_in_smiles2rep:
        print(smi)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"SMILES: {smi} is not valid")
            continue
