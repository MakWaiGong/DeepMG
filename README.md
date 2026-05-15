# DeepMG

DeepMG is a deep learning model for predicting molecule-protein interactions using graph neural networks.

## Project Structure

```
DeepMG/
├── main.py                 # Training script with 5-fold cross-validation
├── models.py               # Model architecture (DeepMG)
├── preprocess.py           # Data preprocessing and graph construction
├── collate_smi.py          # SMILES processing utilities
├── collate_seq.py          # Sequence processing utilities
├── protacs_env.yaml        # Conda environment file
├── .gitignore              # Git ignore file
├── dataset/
│   ├── collate_smi.py
│   ├── collate_seq.py
│   ├── Dataset7.xlsx       # Training data (not in git)
│   ├── embeddings.pkl      # ESM protein embeddings (not in git)
│   ├── smiles2rep.pkl      # Molecule representations (not in git)
│   ├── processed.pkl       # Preprocessed data cache (not in git)
│   └── PDBs/               # Protein PDB files (not in git)
└── pts/                    # Model checkpoints (not in git)
```

## Installation

1. Create conda environment:
```bash
conda env create -f protacs_env.yaml
conda activate protacs_env
```

2. Download large data files from Hugging Face:
```bash
hf upload MakWaiGong/DeepMG dataset/ dataset/
hf upload MakWaiGong/DeepMG pts/ pts/
```

## Usage

Run training with 5-fold cross-validation:
```bash
python main.py
```

## Model Architecture

DeepMG uses a multi-channel graph attention network (GATv2) architecture:

- **DrugGNN**: Processes molecular graphs with 512-dimensional features
- **TargetGNN**: Processes protein structure graphs with 2560-dimensional ESM embeddings
- **DeepMG**: Combines drug and protein representations for interaction prediction

## Requirements

- PyTorch
- PyTorch Geometric
- RDKit
- Biopython
- scikit-learn
- pandas
- numpy
