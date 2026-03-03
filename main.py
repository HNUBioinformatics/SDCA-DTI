#!/usr/bin/env python3
"""
Drug-Protein Interaction Prediction Model
5-fold Cross-Validation Main Script
"""

import warnings
warnings.filterwarnings('ignore')

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from config import Config
from train import train_model


def check_requirements():
    """Check runtime environment"""
    print("Checking runtime environment...")
    
    # CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("⚠ CUDA not available, using CPU")
    
    # Data files
    required_files = [
        Config.ORIGINAL_DATA_FILE,
        Config.ESM2_EMBEDDING_FILE,
        Config.CHEMBERT_EMBEDDING_FILE
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print("✗ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✓ All required files exist")
    return True


def print_model_info():
    """Print core model information"""
    print("\n" + "=" * 80)
    print("Model Architecture")
    print("=" * 80)
    
    print(f"\nProtein Encoder:")
    print(f"  ESM2(640-dim) → Projection(512-dim) → 3-layer DWC(kernel=3) → Max Pooling")
    print(f"  Using CLS features to modulate each convolution layer output")
    
    print(f"\nDrug Encoder:")
    print(f"  ChemBERT(768-dim) → Projection(512-dim) → 3-layer DWC(kernel=3) → Max Pooling")
    print(f"  Using CLS features to modulate each convolution layer output")
    
    print(f"\nDrug Graph Encoder:")
    print(f"  Atom/Edge Embedding(64-dim) → D-MPNN(3 steps) → Self-Attention(8 heads) → Projection(512-dim)")
    
    print(f"\nInteraction Module:")
    print(f"  Interaction 1: Drug atoms ↔ Protein tokens")
    print(f"  Interaction 2: Drug tokens ↔ Protein tokens")
    print(f"  Gated Fusion → Prediction Head")
    
    print(f"\nContrastive Learning:")
    print(f"  Drug DWC global features vs Drug structure features")
    
    print(f"\nTraining Configuration:")
    print(f"  5-fold CV | Batch: {Config.BATCH_SIZE} | LR: {Config.LEARNING_RATE}")
    print(f"  Max Epochs: {Config.EPOCHS} | Early Stopping Patience: {Config.PATIENCE} | Dropout: {Config.DROPOUT}")


def main():
    """Main function"""
    print("=" * 80)
    print("Drug-Protein Interaction Prediction Model - 5-fold Cross-Validation")
    print("=" * 80)
    
    # Check environment
    if not check_requirements():
        print("\n✗ Environment check failed")
        return
    
    # Print model information
    print_model_info()
    
    print("\n" + "=" * 80)
    print("Starting 5-fold cross-validation training...")
    print("=" * 80)
    
    try:
        cv_results = train_model()
        
        print("\n" + "=" * 80)
        print("🎉 5-fold cross-validation completed!")
        
        # Display final statistics
        if cv_results and 'fold_results' in cv_results:
            fold_results = cv_results['fold_results']
            if fold_results:
                # Calculate statistics
                metric_keys = ['AUC', 'AUPR', 'F1', 'ACC', 'Precision', 'Recall', 'Specificity']
                stats = {}
                for k in metric_keys:
                    vals = [fr['best_metrics'][k] for fr in fold_results if fr['best_metrics'] is not None]
                    if vals:
                        import numpy as np
                        stats[k] = {
                            'mean': float(np.mean(vals)),
                            'std': float(np.std(vals))
                        }
                
                print(f"\nFinal average results:")
                for metric in metric_keys:
                    if metric in stats:
                        mean_val = stats[metric]['mean']
                        std_val = stats[metric]['std']
                        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\nResults saved to: {Config.CV_RESULTS_PATH}")
        print(f"Fold models saved to: {Config.CV_MODELS_DIR}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("⚠ Training interrupted")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"✗ Training error: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()