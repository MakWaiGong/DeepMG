import pandas as pd
import pickle
from rdkit import Chem

# Path configuration
smiles2rep_path = "smiles2rep.pkl"
dataset7_path = "dataset7.xlsx"

# Load smiles2rep.pkl
with open(smiles2rep_path, "rb") as f:
    smiles2rep = pickle.load(f)

# Load dataset7
df = pd.read_excel(dataset7_path)

# Get SMILES list from dataset
smiles_list = df["Smiles"].tolist()

# Check which SMILES are in smiles2rep
not_in_smiles2rep = []
for smi in smiles_list:
    if smi not in smiles2rep:
        not_in_smiles2rep.append(smi)

print(f"Total {len(smiles_list)} SMILES, {len(not_in_smiles2rep)} not in smiles2rep.")
if not_in_smiles2rep:
    print("The following SMILES are not in smiles2rep:")
    for smi in not_in_smiles2rep:
        print(smi)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"SMILES: {smi} is not valid")
            continue
