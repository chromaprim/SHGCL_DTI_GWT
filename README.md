# Requirements
- **Python**: 3.9.21  
- **Core Libraries**:
  - `torch`: 2.5.1 (CUDA 12.4)
  - `dgl`: 0.9.0  
  - `torch-geometric`: 2.6.1  
- **Data Processing**:
  - `numpy`: 2.0.1  
  - `pandas`: 2.2.3  
  - `scipy`: 1.13.1  
  - `scikit-learn`: 1.6.1  

---

# Quick Start
1. Navigate to the source directory:
   ```bash
   cd src/application
   ```
2. Run the main script with optional arguments:
   ```bash
   python main.py [OPTIONS]
   ```

## Command Line Options
| Argument       | Description | Default | Choices |
|----------------|-------------|---------|---------|
| `--device`     | GPU device ID | `cuda:0` | - |
| `--hid_dim`    | Hidden layer dimension | `2048` | - |
| `--number`     | Positive/Negative sample ratio | `ten` | `one` (1:1), `ten` (1:10), `all` (all unlabeled as negative) |
| `--feature`    | Feature type | `default` (ours) | `random`, `luo`, `default` |
| `--task`       | Experimental scenario | `benchmark` | `benchmark`, `disease`, `drug`, `homo_protein_drug`, `sideeffect`, `unique` |
| `--edge_mask`  | Edge masking in HN | `''` (empty) | `drug`, `protein`, `disease`, `sideeffect`, `drugsim`, `proteinsim` (comma-separated) |

---

# Data Description
## Core Files
- **Entity Lists**:
  - `drug.txt`: Drug names  
  - `protein.txt`: Protein names  
  - `disease.txt`: Disease names  
  - `se.txt`: Side effect names  

- **ID Mappings**:
  - `drug_dict_map.txt`: DrugBank ID mappings  
  - `protein_dict_map.txt`: UniProt ID mappings  

## Interaction Matrices
| File | Description |
|------|-------------|
| `mat_drug_se.txt` | Drug-SideEffect associations |
| `mat_protein_protein.txt` | Protein-Protein interactions |
| `mat_drug_drug.txt` | Drug-Drug interactions |
| `mat_protein_disease.txt` | Protein-Disease associations |
| `mat_drug_disease.txt` | Drug-Disease associations |
| `mat_protein_drug.txt` | Protein-Drug interactions |
| `mat_drug_protein.txt` | Drug-Protein interactions |

## Similarity Matrices
- `Similarity_Matrix_Drugs.txt`: Chemical structure-based drug similarity  
- `Similarity_Matrix_Proteins.txt`: Protein sequence similarity  
