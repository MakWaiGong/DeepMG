import os
import pandas as pd
import pickle
from Bio.PDB import PDBParser
from tqdm import tqdm
# 配置路径
xlsx_path = "Dataset7.xlsx"
pdb_dir = "PDBs"
esm_embed_path = "embeddings.pkl"

# 加载 Excel
df = pd.read_excel(xlsx_path)

# 加载 ESM embedding
id2repr = pickle.load(open(esm_embed_path, "rb"))

parser = PDBParser(QUIET=True)

# 记录不匹配的条目
mismatch_list = []

for idx, row in tqdm(df.iterrows(),total=len(df),desc="Checking sequences"):
    seq_id = row["Sequence_ID"]
    effector_id = row["Effector_Sequence_ID"]
    seq = row["Sequence"]
    effector_seq = row["Effector Sequence"]
    
    for protein_type, pid, protein_seq in [("Sequence", seq_id, seq), ("Effector Sequence", effector_id, effector_seq)]:
        if pid not in id2repr:
            mismatch_list.append({
                "ID": row["ID"],
                "Type": protein_type,
                "Sequence_ID": pid,
                "Issue": "Missing ESM embedding"
            })
            continue

        esm_tokens = id2repr[pid]["token_representations"]
        esm_len = len(esm_tokens)
        seq_len = len(protein_seq)
        issue = None

        # 检查序列长度与 ESM token 长度
        if seq_len != esm_len:
            issue = f"Seq_len != ESM_len ({seq_len} vs {esm_len})"

        # 检查 PDB 文件
        pdb_file = os.path.join(pdb_dir, f"{pid}.pdb")
        if not os.path.exists(pdb_file):
            if issue:
                issue += " | "
            else:
                issue = ""
            issue += "Missing PDB file"
        else:
            try:
                structure = parser.get_structure(pid, pdb_file)
                residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
                pdb_len = len(residues)
                if pdb_len != esm_len:
                    if issue:
                        issue += " | "
                    else:
                        issue = ""
                    issue += f"PDB_len != ESM_len ({pdb_len} vs {esm_len})"
            except Exception as e:
                if issue:
                    issue += " | "
                else:
                    issue = ""
                issue += f"PDB parse error: {e}"

        # 如果有问题就记录
        if issue:
            mismatch_list.append({
                "ID": row["ID"],
                "Type": protein_type,
                "Sequence_ID": pid,
                "Seq_len": seq_len,
                "ESM_len": esm_len,
                "Issue": issue
            })

# 输出结果
if mismatch_list:
    mismatch_df = pd.DataFrame(mismatch_list)
    print("发现以下异常条目：")
    print(mismatch_df)
else:
    print("所有序列、ESM token 和 PDB 残基长度均匹配。")


