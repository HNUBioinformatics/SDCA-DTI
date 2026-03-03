from pathlib import Path
import torch

class Config:
    # Data paths
    DATA_ROOT = Path("/home/u2308283088/libiao")

    # Original dataset path (for 5-fold cross-validation)
    ORIGINAL_DATA_FILE = DATA_ROOT / "dataset1/human_cleaned.csv"

    # Pre-trained feature paths
    ESM2_EMBEDDING_FILE = DATA_ROOT / "esm2_features/human_esm2_embeddings.pt"
    CHEMBERT_EMBEDDING_FILE = DATA_ROOT / "chembert_features/human_chembert_embeddings.pt"

    # 5-fold cross-validation model and result paths
    CV_MODELS_DIR = DATA_ROOT / "2026_1_28/best_code_change2/human/cv_models"
    CV_RESULTS_PATH = DATA_ROOT / "2026_1_28/best_code_change2/human/cv_results.pt"

    # Cross-validation parameters
    N_FOLDS = 5

    # Resume training
    RESUME_TRAINING = False
    RESUME_FOLD = 0  # Start from which fold (0-4)

    # Random seed
    RANDOM_SEED = 42

    # Sequence length
    MAX_DRUG_LENGTH = 256
    MAX_PROTEIN_LENGTH = 1024

    # Pre-trained feature dimensions
    ESM2_DIM = 640
    CHEMBERT_DIM = 768

    # Protein module parameters (single-scale)
    PROTEIN_CNN_KERNEL = 3
    PROTEIN_CNN_LAYERS = 3
    PROTEIN_TOKEN_DIM = 512

    # Drug CNN parameters (DWC)
    DRUG_TOKEN_DIM = 512
    DRUG_CNN_LAYERS = 3
    DRUG_CNN_KERNEL = 3

    # Drug graph parameters
    ATOM_FEAT_DIM = 78
    BOND_FEAT_DIM = 12
    GRAPH_NODE_DIM = 64
    DMPNN_HIDDEN_DIM = 64
    DMPNN_STEPS = 3
    ATOM_FINAL_DIM = 512

    # Cross-attention parameters
    CROSS_ATTN_DIM = 512
    
    # ==================== 新增：多头注意力配置 ====================
    NUM_ATTENTION_HEADS = 8  # 注意力头数

    # Final feature dimension
    FINAL_DIM = 512

    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    EPOCHS = 80
    PATIENCE = 20
    DROPOUT = 0.3

    # Contrastive learning
    CONTRASTIVE_TEMP = 0.1
    CONTRASTIVE_WEIGHT = 0.1

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')