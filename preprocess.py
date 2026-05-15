import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Data
from rdkit.Chem.rdchem import BondType
import json
import pickle
import os
import gc
import requests
from Bio.PDB.PDBParser import PDBParser
import random
from sklearn.model_selection import train_test_split, KFold
from rdkit.Chem import QED

from torch import nn
import psutil
from tqdm import tqdm
tqdm.pandas()

BOND_TYPE = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def smis2graphs(smis):
    smi2repr = pickle.load(open(os.path.join("dataset","smiles2rep.pkl"), "rb"))
    graphs = {}
    for smi in tqdm(smis,desc="smis2graphs"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[smis2graphs]Invalid SMILES string: {smi}")
            continue

        mol = Chem.AddHs(mol)
        graph = Data()
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()

        if smi not in smi2repr:
            print(f"[smis2graphs]SMILES not found in representation: {smi}")
            continue
        graph.x = torch.Tensor(smi2repr[smi]["atomic_reprs"][0])
        if graph.x.shape != (len(atoms), 512):
            print('[smis2graphs]',smi, graph.x.shape, len(atoms))
            graphs[smi] = None
            continue

        edge_index = []
        edge_attr = []
        for bond in bonds:
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
            edge_attr.append(
                one_of_k_encoding(bond.GetBondType(), BOND_TYPE)
                + [bond.GetIsAromatic(), bond.GetIsConjugated(), bond.IsInRing()]
            )
            edge_attr.append(
                one_of_k_encoding(bond.GetBondType(), BOND_TYPE)
                + [bond.GetIsAromatic(), bond.GetIsConjugated(), bond.IsInRing()]
            )
        graph.edge_index = torch.LongTensor(np.transpose(edge_index))
        if graph.edge_index.shape != (2, len(bonds) * 2):
            print(f"[smis2graphs] Wrong shape of graph.edge_index for {smi}: {graph.edge_index.shape}, expected (2, {len(bonds) * 2})")
            continue
        graph.edge_attr = torch.Tensor(edge_attr)
        if graph.edge_attr.shape != (len(bonds) * 2, 7):
            print(f"[smis2graphs] Wrong shape of graph.edge_attr for {smi}: {graph.edge_attr.shape}, expected ({len(bonds) * 2}, 7)")
            continue
        graphs[smi] = graph
    return graphs


def seqs2graphs(ids_seqs):

    def distance_map(id, seq):
        DISTANCE_CUTOFF = 8.0
        parser = PDBParser()
        with open(os.path.join("dataset","PDBs", str(id) + ".pdb"), "r") as f:
            structure = parser.get_structure(id, f)
        residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
        edge_index = []
        edge_weight = []
        for i in range(len(residues)):
            for j in range(len(residues)):
                if i != j:
                    distance = np.linalg.norm(
                        residues[i]["CA"].get_coord() - residues[j]["CA"].get_coord()
                    )
                    if distance < DISTANCE_CUTOFF:
                        edge_index.append([i, j])
                        edge_weight.append(distance)

        edge_weight = np.asarray(edge_weight)
        edge_weight = (edge_weight.max() - edge_weight) / (np.ptp(edge_weight))
        return (
            torch.LongTensor(np.transpose(edge_index)),
            torch.Tensor(edge_weight).unsqueeze(1),
        )

    id2repr = pickle.load(open(os.path.join("dataset", "embeddings.pkl"), "rb"))
    graphs = {}
    for id, seq in ids_seqs.items():
        graph = Data()
        assert "id2repr" in locals() and id in id2repr
        graph.x = torch.Tensor(id2repr[id]["token_representations"])
        assert graph.x.shape[1] == 2560

        graph.edge_index, graph.edge_attr = distance_map(id, seq)
        assert graph.edge_index.shape == (2, len(graph.edge_attr))
        graphs[id] = graph
    return graphs



def seq2graph_with_prompt(id, seq, prompt_feat, id2repr,proj_layer=nn.Linear(8, 2560)):

    def distance_map(id, seq, pdb_dir="dataset/PDBs"):
        DISTANCE_CUTOFF = 8.0
        parser = PDBParser()
        with open(os.path.join(pdb_dir, str(id) + ".pdb"), "r") as f:
            structure = parser.get_structure(id, f)
        residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]

        edge_index = []
        edge_weight = []
        for i in range(len(residues)):
            for j in range(len(residues)):
                if i != j and "CA" in residues[i] and "CA" in residues[j]:
                    distance = np.linalg.norm(
                        residues[i]["CA"].get_coord() - residues[j]["CA"].get_coord()
                    )
                    if distance < DISTANCE_CUTOFF:
                        edge_index.append([i, j])
                        edge_weight.append(distance)

        edge_weight = np.asarray(edge_weight)
        edge_weight = (edge_weight.max() - edge_weight) / (np.ptp(edge_weight))
        return (
            torch.LongTensor(np.transpose(edge_index)),
            torch.Tensor(edge_weight).unsqueeze(1),
        )


    graph = Data()
    # Node feature from ESM embedding
    assert id in id2repr
    x = torch.Tensor(id2repr[id]["token_representations"])
    assert x.shape[1] == 2560

    # Prompt embedding
    if isinstance(prompt_feat, np.ndarray):
        prompt_feat = torch.tensor(prompt_feat)
    if prompt_feat.dim() == 1:
        prompt_feat = prompt_feat.unsqueeze(0)

    if proj_layer is not None:
        prompt_feat = proj_layer(prompt_feat.to(proj_layer.weight.device))

    # Append prompt as a new node
    x = torch.cat([x, prompt_feat], dim=0)
    x = x.clone().detach()

    # Build edges
    edge_index, edge_attr = distance_map(id, seq)
    L = x.shape[0] - 1
    prompt_edges = [[L, i] for i in range(L)] + [[i, L] for i in range(L)]
    prompt_edges = torch.LongTensor(prompt_edges).T
    prompt_attr = torch.ones((prompt_edges.shape[1], 1))

    # Combine edges
    edge_index = torch.cat([edge_index, prompt_edges], dim=1)
    edge_attr = torch.cat([edge_attr, prompt_attr], dim=0)

    graph.x = x
    graph.edge_index = edge_index
    graph.edge_attr = edge_attr
    graph.batch = torch.zeros(x.shape[0], dtype=torch.long)
    return graph

def seq2graph_with_prompt_save(id, seq, prompt_feat, id2repr, proj_layer=None, pdb_dir="dataset/PDBs", distance_cutoff=8.0):

    def distance_map(id, seq, pdb_dir=pdb_dir, distance_cutoff=distance_cutoff):
        parser = PDBParser()
        with open(os.path.join(pdb_dir, str(id) + ".pdb"), "r") as f:
            structure = parser.get_structure(id, f)
        residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]

        edge_index = []
        edge_weight = []
        for i in range(len(residues)):
            for j in range(len(residues)):
                if i != j and "CA" in residues[i] and "CA" in residues[j]:
                    distance = np.linalg.norm(
                        residues[i]["CA"].get_coord() - residues[j]["CA"].get_coord()
                    )
                    if distance < distance_cutoff:
                        edge_index.append([i, j])
                        edge_weight.append(distance)

        if len(edge_index) == 0:
            return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,1))
        edge_index = np.array(edge_index).T
        edge_weight = np.array(edge_weight)
        edge_weight = (edge_weight.max() - edge_weight) / (np.ptp(edge_weight) + 1e-6)
        return torch.LongTensor(edge_index), torch.Tensor(edge_weight).unsqueeze(1)

    graph = Data()
    # Node feature from ESM embedding
    assert id in id2repr, f"{id} not in id2repr"
    x = torch.Tensor(id2repr[id]["token_representations"])
    L_token = x.shape[0]

    # Prompt embedding
    if isinstance(prompt_feat, np.ndarray):
        prompt_feat = torch.tensor(prompt_feat)
    if prompt_feat.dim() == 1:
        prompt_feat = prompt_feat.unsqueeze(0)

    if proj_layer is not None:
        prompt_feat = proj_layer(prompt_feat.to(next(proj_layer.parameters()).device))

    # Append prompt as a new node
    x = torch.cat([x, prompt_feat], dim=0)

    # Build edges
    edge_index, edge_attr = distance_map(id, seq)
    # Safe handling: clip edges if residue count < token count
    L_res = x.shape[0] - 1
    if edge_index.numel() > 0:
        mask = (edge_index[0] < L_res) & (edge_index[1] < L_res)
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask]

    # Add prompt edges
    prompt_edges = [[L_res, i] for i in range(L_res)] + [[i, L_res] for i in range(L_res)]
    prompt_edges = torch.LongTensor(prompt_edges).T
    prompt_attr = torch.ones((prompt_edges.shape[1], 1))

    # Combine edges
    edge_index = torch.cat([edge_index, prompt_edges], dim=1)
    edge_attr = torch.cat([edge_attr, prompt_attr], dim=0)

    # Final sanity check
    assert edge_index.max().item() < x.shape[0], f"{id} edge_index max {edge_index.max()} >= num_nodes {x.shape[0]}"
    assert edge_index.min().item() >= 0, f"{id} edge_index min {edge_index.min()} < 0"
    assert edge_attr.shape[0] == edge_index.shape[1], f"{id} edge_attr.size(0) != edge_index.size(1)"

    graph.x = x
    graph.edge_index = edge_index
    graph.edge_attr = edge_attr
    graph.batch = torch.zeros(x.shape[0], dtype=torch.long)
    return graph

# Calculate molecular descriptors
def calculate_molecule_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    qed = QED.properties(mol)

    descriptors = {
        "MW": qed.MW,
        "ALOGP":qed.ALOGP,
        "HBA":qed.HBA,
        "HBD":qed.HBD,
        "PSA":qed.PSA,
        "ROTB":qed.ROTB,
        "AROM":qed.AROM,
        "ALERTS":qed.ALERTS
    }

    return np.array(list(descriptors.values()))

aa_properties = {
    'A': [1.28, 0.05, 88.6],   # Alanine
    'C': [0.77, -0.10, 108.5], # Cysteine
    'D': [-1.05, 0.80, 111.1], # Aspartic Acid
    'E': [-1.01, 0.79, 138.4], # Glutamic Acid
    'F': [1.22, -0.56, 189.9], # Phenylalanine
    'G': [0.00, 0.00, 60.1],   # Glycine
    'H': [0.96, 0.26, 153.2],  # Histidine
    'I': [1.31, -0.60, 166.7], # Isoleucine
    'K': [-0.99, 0.97, 168.6], # Lysine
    'L': [1.21, -0.57, 166.7], # Leucine
    'M': [1.27, -0.23, 162.9], # Methionine
    'N': [-0.60, 0.85, 114.1], # Asparagine
    'P': [0.72, 0.00, 112.7],  # Proline
    'Q': [-0.22, 0.85, 143.8], # Glutamine
    'R': [-0.99, 0.95, 173.4], # Arginine
    'S': [-0.84, 0.52, 89.0],  # Serine
    'T': [-0.27, 0.27, 116.1], # Threonine
    'V': [1.00, -0.54, 140.0], # Valine
    'W': [0.88, -0.24, 227.8], # Tryptophan
    'Y': [0.33, 0.02, 193.6],  # Tyrosine
}

# Extract ACC features from protein sequences
def extract_acc_features(sequence, max_lag=6):
    """
    Extract ACC (Auto-Cross Covariance) features from protein sequences.

    Args:
        sequence (str): Protein amino acid sequence.
        max_lag (int): Maximum lag value, default 6.

    Returns:
        np.ndarray: Generated ACC feature vector.
    """

    def get_property_matrix(sequence, properties):
        matrix = []
        for aa in sequence:
            if aa in properties:
                matrix.append(properties[aa])
            else:
                matrix.append([0] * len(next(iter(properties.values()))))
        return np.array(matrix)

    def calculate_acc(property_matrix, max_lag):
        num_props, seq_len = property_matrix.shape[1], property_matrix.shape[0]
        acc_features = []
        for k in range(num_props):
            prop_values = property_matrix[:, k]
            mean_k = np.mean(prop_values)
            for tau in range(1, max_lag + 1):
                if seq_len > tau:
                    cov = np.mean((prop_values[:seq_len - tau] - mean_k) *
                                  (prop_values[tau:] - mean_k))
                    acc_features.append(cov)
                else:
                    acc_features.append(0)
        return np.array(acc_features)

    property_matrix = get_property_matrix(sequence, aa_properties)
    acc_features = calculate_acc(property_matrix, max_lag)
    return acc_features




def combine_descriptors(smiles, seq_a, seq_b):
    mol_desc = calculate_molecule_descriptors(smiles)
    prot_a_desc = extract_acc_features(seq_a)
    prot_b_desc = extract_acc_features(seq_b)
    if np.any(mol_desc) and np.any(prot_a_desc) and np.any(prot_b_desc):
        return np.concatenate([mol_desc, prot_a_desc, prot_b_desc])
    else:
        return None






def load_data(params):
    fold = params["fold"] if "fold" in params else None
    valid = params["valid"]
    seed = params['seed']


    if valid:
        assert fold in [0, 1, 2, 3, 4]

    if os.access(os.path.join("dataset", "processed.pkl"), os.R_OK):
        print("[INFO] Loading cached data.")
        with open(os.path.join("dataset", "processed.pkl"), 'rb') as file:
            data = pickle.load(file)
        print(f"[INFO] Loaded {len(data)} samples from cache.")
    else:
        print(f"Cache not found, processing data...")
        # Step 1: Load full.csv
        full = pd.read_excel(os.path.join("dataset", "Dataset7.xlsx"))
        # Step 2: Build protein sequence dict (using Uniprot ID as key)
        seq_dict = {}
        for _, row in full.iterrows():
            seq_dict[row["Effector_Sequence_ID"]] = row["Effector Sequence"]
            seq_dict[row["Sequence_ID"]] = row["Sequence"]
        prot_graphs = seqs2graphs(seq_dict)
        print("Graphs finished!")

        # Step 4: Build small molecule graphs
        drug_graphs = smis2graphs(set(full["Smiles"].tolist()))
        print("Drug graphs finished!")


        data = []
        for _, row in full.iterrows():
            e3_id = row["Effector_Sequence_ID"]
            poi_id = row["Sequence_ID"]
            e3_seq = row["Effector Sequence"]
            poi_seq = row["Sequence"]
            smiles = row["Smiles"]
            label = row["label"]
            smiles_id = row["Smiles_ID"]


            if drug_graphs[smiles] and prot_graphs[e3_id] and  prot_graphs[poi_id] and not np.isnan(label):
                data.append([
                        drug_graphs[smiles],
                        prot_graphs[e3_id],
                        prot_graphs[poi_id],
                        torch.Tensor([label])
                    ])
            else:
                print("[Warning] Skipping sample due to missing or invalid input:")
                print(f"Current sample ID: e3:{e3_id}, poi:{poi_id}, smiles:{smiles_id}\n")
        print(f"Processing finished, length: {len(data)}")
        pickle.dump(data, open(os.path.join("dataset", "processed.pkl"), "wb"))


    # Step 7: Split train/valid/test
    for i, d in enumerate(data):
        if d is None:
            print(f"[DEBUG] None found at index {i}")
    random.shuffle(data)
    tra_data, test_data = train_test_split(data, test_size=0.1, random_state=seed)
    chunk_size = len(tra_data) // 5
    start_idx = fold * chunk_size
    end_idx = len(tra_data) if fold == 4 else start_idx + chunk_size
    valid_data = tra_data[start_idx:end_idx]
    train_data = tra_data[:start_idx] + tra_data[end_idx:]

    return train_data, valid_data, test_data
