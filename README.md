# SDCA-DTI Model

A deep learning model for drug-protein interaction prediction using multi-modal features and collaborative attention mechanisms with 5-fold cross-validation.

## Setup and Dependencies

Dependencies:

- python 3.10+
- pytorch >=2.3
- torch-geometric
- pandas
- numpy
- scikit-learn
- rdkit
- tqdm

## Project Structure

```
├── README.md                  # This file
├── config.py                  # Configuration and hyperparameters
├── data_processing.py         # Dataset and data loaders
├── model.py                   # Model architecture
├── train.py                   # Training and evaluation functions
├── main.py                    # Main entry point
├── cleaned.py                 # Data cleaning utility
└── dataset/                   # Dataset directory (Raw data)
```

## Dataset Format

The dataset should be a CSV file with the following columns:

```
smiles,sequence,label
COC1=C(C=C2C(=C...,MTVKTEAAKGTLTYSRMRGM...,1
```

## Pre-trained Features

Before training, you need to prepare pre-trained embeddings:

- **ESM2 embeddings**: Protein sequence features and [CLS] embeddings
- **ChemBERT embeddings**: Drug SMILES features and [CLS] embeddings

pertrained model download

##### https://huggingface.co/facebook/esm2_t33_650M_UR50D

##### https://huggingface.co/seyonec/ChemBERTa-zinc-base-v1

Place the embedding files in the paths specified in `config.py`:

- `ESM2_EMBEDDING_FILE`: ESM2 protein embeddings
- `CHEMBERT_EMBEDDING_FILE`: ChemBERT drug embeddings 

## Run

### step1: Data Cleaning (Optional)

```bash
python cleaned.py
```

This removes compounds with salts/mixtures (SMILES containing '.').

### step2: Extract Pretrained Embeddings

##### exact the  protein embeddings :

```bash
python extract_esm2.py
```

##### exact the drug embeddings:

```bash
python extract_chemberta.py
```

### step3: Training

```bash
python main.py
```