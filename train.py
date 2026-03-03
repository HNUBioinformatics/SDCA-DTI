import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from tqdm import tqdm
from config import Config
from data_processing import prepare_data
from model import DrugProteinInteractionModel


def calculate_metrics(y_true, y_pred, y_scores):
    """Calculate evaluation metrics"""
    if len(np.unique(y_true)) < 2:
        return {
            'AUC': 0.5, 'AUPR': float(np.mean(y_true)), 'F1': 0.0,
            'ACC': float(np.mean(y_pred == y_true)), 'Precision': 0.0,
            'Recall': 0.0, 'Specificity': 0.0
        }

    auc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    f1 = f1_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'AUC': auc, 'AUPR': aupr, 'F1': f1, 'ACC': acc,
        'Precision': precision, 'Recall': recall, 'Specificity': specificity
    }


def save_checkpoint(epoch, model, optimizer, best_auc, best_epoch,
                    best_metrics, best_model_state, history,
                    patience_counter, checkpoint_path):
    """Save checkpoint with complete training state"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_auc': best_auc,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
        'best_model_state': best_model_state,
        'history': history,
        'patience_counter': patience_counter,
        'random_state': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer):
    """Load checkpoint and restore training state"""
    if not checkpoint_path.exists():
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        return None

    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore random states
    if 'random_state' in checkpoint:
        try:
            torch_rng_state = checkpoint['random_state']['torch']
            if torch_rng_state is not None:
                if not isinstance(torch_rng_state, torch.ByteTensor):
                    torch_rng_state = torch_rng_state.byte()
                torch_rng_state = torch_rng_state.cpu()
                torch.set_rng_state(torch_rng_state)

            if checkpoint['random_state']['numpy'] is not None:
                np.random.set_state(checkpoint['random_state']['numpy'])

            if torch.cuda.is_available() and checkpoint['random_state']['cuda'] is not None:
                cuda_states = checkpoint['random_state']['cuda']
                if cuda_states is not None:
                    cuda_states = [s.byte().cpu() if not isinstance(s, torch.ByteTensor) else s
                                   for s in cuda_states]
                    torch.cuda.set_rng_state_all(cuda_states)

            print("✓ Random states restored")
        except Exception as e:
            print(f"⚠ Failed to restore random states: {e}")

    print(f"✓ Resuming from Epoch {checkpoint['epoch'] + 1}")
    print(f"  Best AUC: {checkpoint['best_auc']:.4f} (Epoch {checkpoint['best_epoch'] + 1})")

    return checkpoint


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_main_loss = 0
    total_contrastive_loss = 0

    pbar = tqdm(train_loader, desc='Training')

    for batch in pbar:
        for key in batch:
            if key != 'drug_graphs':
                batch[key] = batch[key].to(device, non_blocking=True)
        batch['drug_graphs'] = batch['drug_graphs'].to(device, non_blocking=True)

        optimizer.zero_grad()

        logits, contrastive_loss = model(
            batch['drug_graphs'],
            batch['chembert_token_feat'],
            batch['chembert_cls_feat'],
            batch['chembert_token_mask'],
            batch['esm2_token_feat'],
            batch['esm2_cls_feat'],
            batch['esm2_token_mask']
        )

        main_loss = criterion(logits, batch['labels'])
        total_loss = main_loss

        if contrastive_loss is not None:
            total_loss = total_loss + Config.CONTRASTIVE_WEIGHT * contrastive_loss
            current_contrastive_loss = contrastive_loss.item()
        else:
            current_contrastive_loss = 0.0

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_main_loss += main_loss.item()
        total_contrastive_loss += current_contrastive_loss

        pbar.set_postfix({
            'Loss': f'{main_loss.item():.4f}',
            'Contr': f'{current_contrastive_loss:.4f}'
        })

    return {
        'main_loss': total_main_loss / len(train_loader),
        'contrastive_loss': total_contrastive_loss / len(train_loader)
    }


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []

    pbar = tqdm(val_loader, desc='Validation')

    with torch.no_grad():
        for batch in pbar:
            for key in batch:
                if key != 'drug_graphs':
                    batch[key] = batch[key].to(device, non_blocking=True)
            batch['drug_graphs'] = batch['drug_graphs'].to(device, non_blocking=True)

            logits, _ = model(
                batch['drug_graphs'],
                batch['chembert_token_feat'],
                batch['chembert_cls_feat'],
                batch['chembert_token_mask'],
                batch['esm2_token_feat'],
                batch['esm2_cls_feat'],
                batch['esm2_token_mask']
            )

            loss = criterion(logits, batch['labels'])
            total_loss += loss.item()

            scores = torch.sigmoid(logits).cpu().numpy()
            labels = batch['labels'].cpu().numpy()

            all_scores.extend(scores)
            all_labels.extend(labels)

            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    all_preds = (all_scores > 0.5).astype(int)

    metrics = calculate_metrics(all_labels, all_preds, all_scores)
    metrics['loss'] = total_loss / len(val_loader)

    return metrics


def print_metrics(train_metrics, val_metrics, epoch, best_auc, patience_counter):
    """Print all metrics in a formatted way"""
    print(f"\nEpoch {epoch + 1}:")
    print(f"  Train - Loss: {train_metrics['main_loss']:.4f}, Contr: {train_metrics['contrastive_loss']:.4f}")
    print(f"  Val - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['AUC']:.4f}, AUPR: {val_metrics['AUPR']:.4f}")
    print(f"  Val - F1: {val_metrics['F1']:.4f}, ACC: {val_metrics['ACC']:.4f}, Prec: {val_metrics['Precision']:.4f}")
    print(f"  Val - Rec: {val_metrics['Recall']:.4f}, Spec: {val_metrics['Specificity']:.4f}")
    print(f"  Best: {best_auc:.4f}, Pat: {patience_counter}/{Config.PATIENCE}")


def train_model():
    """Train the model with 5-fold cross-validation"""
    print(f"\n{'='*80}")
    print(f"Starting 5-fold cross-validation training")
    print(f"{'='*80}")

    # Set random seeds (unified)
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.RANDOM_SEED)

    # Prepare 5-fold data
    fold_loaders = prepare_data()

    fold_results = []

    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        if fold_idx < Config.RESUME_FOLD:
            print(f"\nSkipping Fold {fold_idx + 1} (already completed or no need to retrain)")
            continue

        print(f"\n{'='*80}")
        print(f"Fold {fold_idx + 1}/{Config.N_FOLDS}")
        print(f"{'='*80}")

        model = DrugProteinInteractionModel().to(Config.DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = -np.inf
        best_epoch = -1
        best_metrics = None
        best_model_state = None
        patience_counter = 0
        start_epoch = 0

        history = {
            'train_loss': [], 'val_loss': [], 'val_auc': [],
            'val_aupr': [], 'val_f1': [], 'val_recall': [], 'contrastive_loss': []
        }

        checkpoint_path = Config.CV_MODELS_DIR / f"fold{fold_idx}_checkpoint.pth"
        best_model_path = Config.CV_MODELS_DIR / f"fold{fold_idx}_best_model.pth"
        history_path = Config.CV_MODELS_DIR / f"fold{fold_idx}_history.pt"

        # Load checkpoint (optional)
        if Config.RESUME_TRAINING and checkpoint_path.exists():
            checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
            if checkpoint is not None:
                best_auc = checkpoint['best_auc']
                best_epoch = checkpoint['best_epoch']
                best_metrics = checkpoint['best_metrics']
                best_model_state = checkpoint['best_model_state']
                history = checkpoint['history']
                patience_counter = checkpoint['patience_counter']
                start_epoch = checkpoint['epoch'] + 1
                print(f"✓ Resuming Fold {fold_idx + 1} from epoch {start_epoch + 1}")
        else:
            if Config.RESUME_TRAINING:
                print(f"⚠ No checkpoint for Fold {fold_idx + 1}, starting from scratch")

        # Training loop
        for epoch in range(start_epoch, Config.EPOCHS):
            print(f"\n{'='*80}")
            print(f"Fold {fold_idx + 1} - Epoch {epoch + 1}/{Config.EPOCHS}")
            print(f"{'='*80}")

            train_metrics = train_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
            val_metrics = validate_epoch(model, val_loader, criterion, Config.DEVICE)

            history['train_loss'].append(train_metrics['main_loss'])
            history['contrastive_loss'].append(train_metrics['contrastive_loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_auc'].append(val_metrics['AUC'])
            history['val_aupr'].append(val_metrics['AUPR'])
            history['val_f1'].append(val_metrics['F1'])
            history['val_recall'].append(val_metrics['Recall'])

            if val_metrics['AUC'] > best_auc:
                best_auc = val_metrics['AUC']
                best_epoch = epoch
                best_metrics = val_metrics
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"\n✓ New best AUC: {best_auc:.4f}")
            else:
                patience_counter += 1

            print_metrics(train_metrics, val_metrics, epoch, best_auc, patience_counter)

            save_checkpoint(
                epoch, model, optimizer, best_auc, best_epoch,
                best_metrics, best_model_state, history, patience_counter,
                checkpoint_path
            )
            print(f"✓ Checkpoint saved: {checkpoint_path}")

            if patience_counter >= Config.PATIENCE:
                print(f"\n⚠ Early stopping triggered on fold {fold_idx + 1}!")
                print(f"   Best AUC: {best_auc:.4f} at Epoch {best_epoch + 1}")
                break

        # Save best model and history
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        Config.CV_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': best_metrics,
            'epoch': best_epoch,
            'fold': fold_idx
        }, best_model_path)
        torch.save(history, history_path)
        print(f"✓ Best model saved: {best_model_path}")
        print(f"✓ History saved: {history_path}")

        # Optional: remove checkpoint for this fold
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"✓ Checkpoint file removed for fold {fold_idx + 1}")

        fold_results.append({
            'fold': fold_idx,
            'best_epoch': best_epoch,
            'best_metrics': best_metrics
        })

    # Aggregate results
    if fold_results:
        # Calculate average metrics
        metric_keys = ['AUC', 'AUPR', 'F1', 'ACC', 'Precision', 'Recall', 'Specificity']
        mean_metrics = {}
        for k in metric_keys:
            vals = [fr['best_metrics'][k] for fr in fold_results if fr['best_metrics'] is not None]
            mean_metrics[k] = float(np.mean(vals)) if len(vals) > 0 else None

        summary = {
            'fold_results': fold_results,
            'mean_metrics': mean_metrics
        }
        Config.CV_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(summary, Config.CV_RESULTS_PATH)
        print(f"\n✓ CV results saved: {Config.CV_RESULTS_PATH}")
        print("Mean metrics:")
        for k, v in mean_metrics.items():
            print(f"  {k}: {v:.4f}" if v is not None else f"  {k}: None")

    print(f"\n✓ 5-fold training completed")
    return {
        'fold_results': fold_results
    }


if __name__ == "__main__":
    train_model()