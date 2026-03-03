import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from torch_geometric.data import Data, Batch
from sklearn.model_selection import StratifiedKFold
from config import Config
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')


class DrugProteinDataset(Dataset):
    def __init__(self, smiles_list, sequences, labels, esm2_embeddings, chembert_embeddings):
        self.smiles_list = smiles_list
        self.sequences = sequences
        self.labels = labels
        self.esm2_embeddings = esm2_embeddings
        self.chembert_embeddings = chembert_embeddings

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Drug graph features
        drug_graph = self.smiles_to_graph(smiles)

        # ChemBERT drug features
        if smiles in self.chembert_embeddings:
            chembert_features = self.chembert_embeddings[smiles]
            raw_cls_feat = chembert_features['global']
            raw_token_feat = chembert_features['tokens']

            chembert_cls_feat = torch.tensor(raw_cls_feat, dtype=torch.float32)

            if len(raw_token_feat) == 0:
                chembert_token_feat = torch.zeros(Config.MAX_DRUG_LENGTH, Config.CHEMBERT_DIM, dtype=torch.float32)
                chembert_token_mask = torch.zeros(Config.MAX_DRUG_LENGTH, dtype=torch.bool)
            else:
                token_tensor = torch.tensor(raw_token_feat, dtype=torch.float32)
                seq_len = token_tensor.size(0)

                if seq_len > Config.MAX_DRUG_LENGTH:
                    chembert_token_feat = token_tensor[:Config.MAX_DRUG_LENGTH]
                    chembert_token_mask = torch.ones(Config.MAX_DRUG_LENGTH, dtype=torch.bool)
                else:
                    padding_size = Config.MAX_DRUG_LENGTH - seq_len
                    padding = torch.zeros(padding_size, Config.CHEMBERT_DIM, dtype=torch.float32)
                    chembert_token_feat = torch.cat([token_tensor, padding], dim=0)

                    valid_mask = torch.ones(seq_len, dtype=torch.bool)
                    padding_mask = torch.zeros(padding_size, dtype=torch.bool)
                    chembert_token_mask = torch.cat([valid_mask, padding_mask], dim=0)
        else:
            chembert_token_feat = torch.zeros(Config.MAX_DRUG_LENGTH, Config.CHEMBERT_DIM, dtype=torch.float32)
            chembert_cls_feat = torch.zeros(Config.CHEMBERT_DIM, dtype=torch.float32)
            chembert_token_mask = torch.zeros(Config.MAX_DRUG_LENGTH, dtype=torch.bool)

        # ESM2 protein features
        if sequence in self.esm2_embeddings:
            esm2_features = self.esm2_embeddings[sequence]
            raw_cls_feat = esm2_features['global']
            raw_token_feat = esm2_features['tokens']

            esm2_cls_feat = torch.tensor(raw_cls_feat, dtype=torch.float32)

            if len(raw_token_feat) == 0:
                esm2_token_feat = torch.zeros(Config.MAX_PROTEIN_LENGTH, Config.ESM2_DIM, dtype=torch.float32)
                esm2_token_mask = torch.zeros(Config.MAX_PROTEIN_LENGTH, dtype=torch.bool)
            else:
                token_tensor = torch.tensor(raw_token_feat, dtype=torch.float32)
                seq_len = token_tensor.size(0)

                if seq_len > Config.MAX_PROTEIN_LENGTH:
                    esm2_token_feat = token_tensor[:Config.MAX_PROTEIN_LENGTH]
                    esm2_token_mask = torch.ones(Config.MAX_PROTEIN_LENGTH, dtype=torch.bool)
                else:
                    padding_size = Config.MAX_PROTEIN_LENGTH - seq_len
                    padding = torch.zeros(padding_size, Config.ESM2_DIM, dtype=torch.float32)
                    esm2_token_feat = torch.cat([token_tensor, padding], dim=0)

                    valid_mask = torch.ones(seq_len, dtype=torch.bool)
                    padding_mask = torch.zeros(padding_size, dtype=torch.bool)
                    esm2_token_mask = torch.cat([valid_mask, padding_mask], dim=0)
        else:
            esm2_token_feat = torch.zeros(Config.MAX_PROTEIN_LENGTH, Config.ESM2_DIM, dtype=torch.float32)
            esm2_cls_feat = torch.zeros(Config.ESM2_DIM, dtype=torch.float32)
            esm2_token_mask = torch.zeros(Config.MAX_PROTEIN_LENGTH, dtype=torch.bool)

        return {
            'drug_graph': drug_graph,
            'chembert_token_feat': chembert_token_feat,
            'chembert_cls_feat': chembert_cls_feat,
            'chembert_token_mask': chembert_token_mask,
            'esm2_token_feat': esm2_token_feat,
            'esm2_cls_feat': esm2_cls_feat,
            'esm2_token_mask': esm2_token_mask,
            'label': torch.tensor(label, dtype=torch.float32),
        }

    def smiles_to_graph(self, smiles):
        """Convert SMILES to molecular graph"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return Data(
                x=torch.zeros(1, Config.ATOM_FEAT_DIM),
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_attr=torch.empty(0, Config.BOND_FEAT_DIM, dtype=torch.float32)
            )

        # Atom features
        atom_features = []
        for atom in mol.GetAtoms():
            feat = self.get_atom_features(atom)
            atom_features.append(feat)

        x = torch.tensor(atom_features, dtype=torch.float32)

        # Edge features
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])

            bond_feat = self.get_bond_features(bond)
            edge_features.extend([bond_feat, bond_feat])

        if len(edge_indices) == 0:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, Config.BOND_FEAT_DIM, dtype=torch.float32)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def get_atom_features(self, atom):
        """Extract 78-dimensional atom features"""
        features = []

        # Atom type (one-hot, 11 dim)
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H']
        symbol = atom.GetSymbol()
        atom_type_feat = [int(symbol == t) for t in atom_types] + [int(symbol not in atom_types)]
        features.extend(atom_type_feat)

        # Atom degree (7 dim)
        degree = atom.GetDegree()
        degree_feat = [int(degree == i) for i in range(7)]
        features.extend(degree_feat)

        # Total hydrogen count (5 dim)
        total_h = atom.GetTotalNumHs()
        h_feat = [int(total_h == i) for i in range(5)]
        features.extend(h_feat)

        # Explicit hydrogen count (5 dim)
        explicit_h = atom.GetNumExplicitHs()
        explicit_h_feat = [int(explicit_h == i) for i in range(5)]
        features.extend(explicit_h_feat)

        # Implicit hydrogen count (5 dim)
        implicit_h = atom.GetNumImplicitHs()
        implicit_h_feat = [int(implicit_h == i) for i in range(5)]
        features.extend(implicit_h_feat)

        # Formal charge (5 dim)
        formal_charge = atom.GetFormalCharge()
        charge_map = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
        charge_feat = [0] * 5
        if formal_charge in charge_map:
            charge_feat[charge_map[formal_charge]] = 1
        elif formal_charge < -2:
            charge_feat[0] = 1
        else:
            charge_feat[4] = 1
        features.extend(charge_feat)

        # Hybridization type (6 dim)
        hybridization = atom.GetHybridization()
        hybrid_types = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]
        hybrid_feat = [int(hybridization == h) for h in hybrid_types] + [int(hybridization not in hybrid_types)]
        features.extend(hybrid_feat)

        # Is aromatic (1 dim)
        features.append(int(atom.GetIsAromatic()))

        # Is in ring (1 dim)
        features.append(int(atom.IsInRing()))

        # Ring size (7 dim)
        ring_size_feat = [0] * 7
        if atom.IsInRing():
            for i in range(3, 9):
                if atom.IsInRingSize(i):
                    ring_size_feat[i-3] = 1
                    break
            else:
                ring_size_feat[6] = 1
        features.extend(ring_size_feat)

        # Total valence (7 dim)
        valence = atom.GetTotalValence()
        valence_feat = [int(valence == i) for i in range(7)]
        features.extend(valence_feat)

        # Radical electrons (5 dim)
        num_radical = atom.GetNumRadicalElectrons()
        radical_feat = [int(num_radical == i) for i in range(5)]
        features.extend(radical_feat)

        # Chirality tag (4 dim)
        chiral_tag = atom.GetChiralTag()
        chiral_types = [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.rdchem.ChiralType.CHI_OTHER
        ]
        chiral_feat = [int(chiral_tag == c) for c in chiral_types]
        features.extend(chiral_feat)

        # Is aromatic and in ring (1 dim)
        features.append(int(atom.GetIsAromatic() and atom.IsInRing()))

        # Atom mass (normalized, 1 dim)
        mass = atom.GetMass() / 100.0
        features.append(mass)

        # Covalent radius (normalized, 1 dim)
        try:
            covalent_radius = Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) / 3.0
        except:
            covalent_radius = 0.0
        features.append(covalent_radius)

        # Van der Waals radius (normalized, 1 dim)
        try:
            vdw_radius = Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) / 3.0
        except:
            vdw_radius = 0.0
        features.append(vdw_radius)

        # Default valence (normalized, 1 dim)
        try:
            default_valence = Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
        except:
            default_valence = 0
        features.append(float(default_valence) / 8.0)

        # Maximum valence (normalized, 1 dim)
        try:
            max_valence = max(Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum()))
        except:
            max_valence = 0
        features.append(float(max_valence) / 8.0)

        # Lone pairs estimation (normalized, 1 dim)
        lone_pairs = max(0, (atom.GetTotalValence() - atom.GetTotalDegree()) // 2)
        features.append(float(lone_pairs) / 4.0)

        # Is connected to heteroatom (1 dim)
        is_connected_hetero = any(neighbor.GetSymbol() not in ['C', 'H'] for neighbor in atom.GetNeighbors())
        features.append(int(is_connected_hetero))

        # Aromatic x ring interaction (1 dim)
        features.append(float(atom.GetIsAromatic()) * float(atom.IsInRing()))

        return features  # Total: 78 dimensions

    def get_bond_features(self, bond):
        """Extract 12-dimensional bond features"""
        features = []

        # Bond type (4 dim)
        bond_type = bond.GetBondType()
        bond_type_feat = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC)
        ]
        features.extend(bond_type_feat)

        # Is in ring (1 dim)
        features.append(int(bond.IsInRing()))

        # Is conjugated (1 dim)
        features.append(int(bond.GetIsConjugated()))

        # Stereo type (6 dim)
        stereo = bond.GetStereo()
        stereo_types = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOANY
        ]
        stereo_feat = [int(stereo == s) for s in stereo_types]
        features.extend(stereo_feat)

        return features  # Total: 12 dimensions


def collate_fn(batch):
    """Custom collate function"""
    drug_graphs = Batch.from_data_list([item['drug_graph'] for item in batch])
    chembert_token_feat = torch.stack([item['chembert_token_feat'] for item in batch])
    chembert_cls_feat = torch.stack([item['chembert_cls_feat'] for item in batch])
    chembert_token_mask = torch.stack([item['chembert_token_mask'] for item in batch])
    esm2_token_feat = torch.stack([item['esm2_token_feat'] for item in batch])
    esm2_cls_feat = torch.stack([item['esm2_cls_feat'] for item in batch])
    esm2_token_mask = torch.stack([item['esm2_token_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    return {
        'drug_graphs': drug_graphs,
        'chembert_token_feat': chembert_token_feat,
        'chembert_cls_feat': chembert_cls_feat,
        'chembert_token_mask': chembert_token_mask,
        'esm2_token_feat': esm2_token_feat,
        'esm2_cls_feat': esm2_cls_feat,
        'esm2_token_mask': esm2_token_mask,
        'labels': labels,
    }


class LazyEmbeddingLoader:
    """Lazy loading of pre-trained features"""
    def __init__(self, file_path, feature_name):
        self.file_path = file_path
        self.feature_name = feature_name
        self._cache = {}
        self._full_data = None
        self._loaded = False

    def __contains__(self, key):
        if not self._loaded:
            self._load_keys_only()
        return key in self._full_data

    def _load_keys_only(self):
        if not self._loaded:
            print(f"Loading {self.feature_name} features: {self.file_path}")
            self._full_data = torch.load(self.file_path, map_location='cpu')
            self._loaded = True
            print(f"{self.feature_name} features loaded, contains {len(self._full_data)} sequences")

    def __getitem__(self, key):
        if not self._loaded:
            self._load_keys_only()

        if key in self._cache:
            return self._cache[key]

        if key in self._full_data:
            features = self._full_data[key]
            if len(self._cache) < 1000:
                self._cache[key] = features
            return features
        else:
            raise KeyError(f"Sequence not found: {key[:50]}...")

    def __len__(self):
        if not self._loaded:
            self._load_keys_only()
        return len(self._full_data)


def prepare_data():
    """Prepare data loaders for 5-fold cross-validation"""
    if not Config.ORIGINAL_DATA_FILE.exists():
        raise FileNotFoundError(f"Original data file not found: {Config.ORIGINAL_DATA_FILE}")

    print(f"\nLoading original dataset: {Config.ORIGINAL_DATA_FILE}")
    df = pd.read_csv(Config.ORIGINAL_DATA_FILE)
    if not {'smiles', 'sequence', 'label'}.issubset(df.columns):
        raise ValueError("Dataset missing required columns: smiles, sequence, label")

    labels = df['label'].values
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS,
        shuffle=True,
        random_state=Config.RANDOM_SEED
    )

    # Lazy load features
    esm2_embeddings = LazyEmbeddingLoader(Config.ESM2_EMBEDDING_FILE, "ESM2")
    chembert_embeddings = LazyEmbeddingLoader(Config.CHEMBERT_EMBEDDING_FILE, "ChemBERT")

    fold_loaders = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, labels)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        print(f"\nFold {fold + 1}/{Config.N_FOLDS}")
        print(f"  Train set: {len(train_df)}  Positive: {train_df['label'].sum()}  Negative: {len(train_df) - train_df['label'].sum()}")
        print(f"  Val set: {len(val_df)}  Positive: {val_df['label'].sum()}  Negative: {len(val_df) - val_df['label'].sum()}")

        train_dataset = DrugProteinDataset(
            train_df['smiles'].tolist(), train_df['sequence'].tolist(),
            train_df['label'].tolist(), esm2_embeddings, chembert_embeddings
        )
        val_dataset = DrugProteinDataset(
            val_df['smiles'].tolist(), val_df['sequence'].tolist(),
            val_df['label'].tolist(), esm2_embeddings, chembert_embeddings
        )

        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
            collate_fn=collate_fn, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )

        fold_loaders.append((train_loader, val_loader))

    print("\n✓ 5-fold data preparation completed")
    return fold_loaders