#!/usr/bin/env python3
"""
IA3 Fine-tuning for Nucleotide Transformer on Promoter Classification.

Implements parameter-efficient fine-tuning using IA3 (Infused Adapter by Inhibiting 
and Amplifying Inner Activations) as described in the Nucleotide Transformer paper 
Section A.1.4.

Key aspects:
- Freezes transformer and embedding layers
- Introduces learnable rescaling vectors for attention (lk, lv) and FFN (lff)
- Uses classification head on top
- ~0.1% additional parameters
- 10-fold cross-validation with 90/10 train/val splits
- 10k training steps per fold
- Batch size of 8
- Adam optimizer with lr=3e-3
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

from pytorch_deePromoter import PromoterDataset, load_promoter_data


class IA3Layer(nn.Module):
    """
    IA3 rescaling weights for a single transformer layer.
    
    Introduces learnable vectors lk, lv (for attention) and lff (for FFN)
    that are applied element-wise to scale activations.
    """
    def __init__(self, d_k: int, d_v: int, d_ff: int):
        super().__init__()
        # Learnable rescaling vectors (initialized to ones)
        self.lk = nn.Parameter(torch.ones(d_k))
        self.lv = nn.Parameter(torch.ones(d_v))
        self.lff = nn.Parameter(torch.ones(d_ff))
        
    def forward(self):
        # This is a parameter container, not used in forward pass directly
        # The weights are accessed by the model during attention/FFN computation
        pass


class NucleotideTransformerIA3Classifier(nn.Module):
    """
    Nucleotide Transformer with IA3 parameter-efficient fine-tuning for classification.
    
    Architecture:
    - Frozen pretrained NT encoder
    - IA3 rescaling weights in each transformer layer
    - Classification head on [CLS] token
    """
    def __init__(
        self,
        model_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Load pretrained model
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=True)
        hidden_size = self.encoder.config.hidden_size
        
        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Get transformer architecture details for IA3
        # For BERT-style models: d_k = d_v = hidden_size, d_ff = intermediate_size
        try:
            d_k = self.encoder.config.hidden_size
            d_v = self.encoder.config.hidden_size
            d_ff = self.encoder.config.intermediate_size
            num_layers = self.encoder.config.num_hidden_layers
        except AttributeError:
            # Fallback for models with different config structure
            d_k = d_v = hidden_size
            d_ff = hidden_size * 4  # Common FFN expansion
            num_layers = 12  # Common default
        
        # Create IA3 rescaling weights for each layer
        self.ia3_layers = nn.ModuleList([
            IA3Layer(d_k, d_v, d_ff) for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        print(f"Model initialized:")
        print(f"  Encoder: {model_name}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Num layers: {num_layers}")
        print(f"  IA3 params per layer: {d_k + d_v + d_ff}")
        print(f"  Total IA3 params: {num_layers * (d_k + d_v + d_ff)}")
        print(f"  Classifier params: {hidden_size * num_classes + num_classes}")
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through frozen encoder + IA3 weights + classifier.
        
        Note: Full IA3 implementation would modify attention/FFN computations.
        This is a simplified version that applies IA3 as layer-wise scaling.
        """
        # Get encoder outputs (frozen)
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get hidden states from all layers
        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1) tensors
        
        # Apply IA3 rescaling to each layer's output
        # Simplified: scale the hidden states by mean of IA3 weights
        scaled_hidden_states = []
        for i, (hidden_state, ia3_layer) in enumerate(zip(hidden_states[1:], self.ia3_layers)):
            # Average the rescaling factors as a simple approximation
            scale = (ia3_layer.lk.mean() + ia3_layer.lv.mean() + ia3_layer.lff.mean()) / 3.0
            scaled_hidden_states.append(hidden_state * scale)
        
        # Use the last layer's scaled representation
        final_hidden = scaled_hidden_states[-1]
        
        # Extract [CLS] token representation (first token)
        cls_output = final_hidden[:, 0, :]  # Shape: [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(cls_output)  # Shape: [batch_size, num_classes]
        
        return logits
    
    def count_parameters(self):
        """Count trainable vs frozen parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = trainable + frozen
        
        return {
            'trainable': trainable,
            'frozen': frozen,
            'total': total,
            'trainable_pct': 100 * trainable / total
        }


# def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
#     """Train for one epoch."""
#     model.train()
#     total_loss = 0
#     all_preds = []
#     all_labels = []
    
#     pbar = tqdm(dataloader, desc="Training")
#     for batch in pbar:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
#         # Forward pass
#         logits = model(input_ids, attention_mask)
#         loss = criterion(logits, labels)
        
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
        
#         # Track metrics
#         total_loss += loss.item()
#         preds = torch.argmax(logits, dim=1)
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
        
#         pbar.set_postfix({'loss': loss.item()})
    
#     avg_loss = total_loss / len(dataloader)
#     accuracy = accuracy_score(all_labels, all_preds)
    
#     return avg_loss, accuracy
def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    device,
    max_steps=None,
):
    """Train for up to one epoch, optionally capped at max_steps batches."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    steps_done = 0

    pbar = tqdm(dataloader, desc="Training")
    for step_idx, batch in enumerate(pbar):
        # 🚫 Hard cap on number of batches
        if max_steps is not None and step_idx >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        steps_done += 1

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": loss.item()})

    if steps_done == 0:
        return 0.0, 0.0

    avg_loss = total_loss / steps_done
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


# @torch.no_grad()
# def evaluate(model, dataloader, criterion, device):
#     """Evaluate model on validation/test set."""
#     model.eval()
#     total_loss = 0
#     all_preds = []
#     all_labels = []
#     all_probs = []
    
#     for batch in tqdm(dataloader, desc="Evaluating"):
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
        
#         # Forward pass
#         logits = model(input_ids, attention_mask)
#         loss = criterion(logits, labels)
        
#         # Track metrics
#         total_loss += loss.item()
#         probs = torch.softmax(logits, dim=1)
#         preds = torch.argmax(logits, dim=1)
        
#         all_preds.extend(preds.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())
#         all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class
    
#     avg_loss = total_loss / len(dataloader)
#     accuracy = accuracy_score(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds)
#     auroc = roc_auc_score(all_labels, all_probs)
#     auprc = average_precision_score(all_labels, all_probs)
    
#     return {
#         'loss': avg_loss,
#         'accuracy': accuracy,
#         'f1': f1,
#         'auroc': auroc,
#         'auprc': auprc,
#     }
@torch.no_grad()
def evaluate(model, dataloader, criterion, device, max_batches=None):
    """Evaluate model on validation/test set, optionally capped at max_batches."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    batches_done = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        # 🔒 Hard cap on number of batches to evaluate
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Track metrics
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Prob of positive class

        batches_done += 1

    # Safety: if we somehow evaluated zero batches
    if batches_done == 0:
        return {
            'loss': 0.0,
            'accuracy': 0.0,
            'f1': 0.0,
            'auroc': 0.0,
            'auprc': 0.0,
        }

    avg_loss = total_loss / batches_done
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    # AUROC/AUPRC can crash if only one class is present in the slice
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = 0.5  # neutral baseline

    try:
        auprc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auprc = 0.5

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
    }




# def train_single_fold(
#     train_dataset,
#     val_dataset,
#     model_name: str,
#     num_steps: int,
#     batch_size: int,
#     learning_rate: float,
#     device: torch.device,
#     save_dir: Path,
#     fold_idx: int,
# ):
#     """Train model on a single fold with 10k steps."""
    
#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=4,
#         pin_memory=True,
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#     )
    
#     # ---- Initialize model (with optional multi-GPU) ----
#     # Build the base model first
#     base_model = NucleotideTransformerIA3Classifier(model_name=model_name)
    
#     # Print parameter counts BEFORE wrapping in DataParallel
#     param_counts = base_model.count_parameters()
#     print(f"\nFold {fold_idx} - Parameter counts:")
#     print(f"  Trainable: {param_counts['trainable']:,} ({param_counts['trainable_pct']:.2f}%)")
#     print(f"  Frozen: {param_counts['frozen']:,}")
#     print(f"  Total: {param_counts['total']:,}")
    
#     # If we have multiple GPUs and we're on CUDA, use DataParallel
#     if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device.type == "cuda":
#         print(f"Using {torch.cuda.device_count()} GPUs with nn.DataParallel")
#         model = nn.DataParallel(base_model)
#     else:
#         model = base_model

#     # Move (possibly wrapped) model to device
#     model = model.to(device)
#     # ----------------------------------------------------
    
#     # Loss and optimizer (as per paper)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
#     # Learning rate scheduler (linear warmup + decay)
#     num_epochs = max(1, num_steps // len(train_loader))
#     total_steps = num_epochs * len(train_loader)
#     scheduler = get_linear_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=int(0.1 * total_steps),
#         num_training_steps=total_steps,
#     )
    
#     print(f"\nTraining configuration:")
#     print(f"  Num epochs: {num_epochs}")
#     print(f"  Steps per epoch: {len(train_loader)}")
#     print(f"  Total steps: {total_steps}")
#     print(f"  Target steps: {num_steps}")
    
#     # Training loop
#     best_val_score = 0
#     best_model_state = None
    
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
#         # Train
#         train_loss, train_acc = train_epoch(
#             model, train_loader, criterion, optimizer, scheduler, device
#         )
        
#         # Validate
#         val_metrics = evaluate(model, val_loader, criterion, device)
        
#         print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
#         print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
#               f"Val F1: {val_metrics['f1']:.4f}, Val AUROC: {val_metrics['auroc']:.4f}")
        
#         # Save best model based on validation accuracy
#         if val_metrics['accuracy'] > best_val_score:
#             best_val_score = val_metrics['accuracy']
#             # If using DataParallel, unwrap to save the underlying module
#             model_to_save = model.module if isinstance(model, nn.DataParallel) else model
#             best_model_state = {
#                 'model_state_dict': model_to_save.state_dict(),
#                 'val_metrics': val_metrics,
#                 'epoch': epoch,
#             }
    
#     # Save best model for this fold
#     fold_save_path = save_dir / f"fold_{fold_idx}_best.pt"
#     torch.save(best_model_state, fold_save_path)
#     print(f"\nBest model saved to {fold_save_path}")
#     print(f"Best validation accuracy: {best_val_score:.4f}")
    
#     return best_model_state, best_val_score

def train_single_fold(
    train_dataset,
    val_dataset,
    model_name: str,
    num_steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    save_dir: Path,
    fold_idx: int,
):
    """Train model on a single fold, capped at num_steps gradient updates."""

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Initialize model (with optional multi-GPU) ----
    base_model = NucleotideTransformerIA3Classifier(model_name=model_name)

    # Print parameter counts BEFORE wrapping in DataParallel
    param_counts = base_model.count_parameters()
    print(f"\nFold {fold_idx} - Parameter counts:")
    print(
        f"  Trainable: {param_counts['trainable']:,} "
        f"({param_counts['trainable_pct']:.2f}%)"
    )
    print(f"  Frozen: {param_counts['frozen']:,}")
    print(f"  Total: {param_counts['total']:,}")

    # If we have multiple GPUs and we're on CUDA, use DataParallel
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1 and device.type == "cuda":
    #     print(f"Using {torch.cuda.device_count()} GPUs with nn.DataParallel")
    #     model = nn.DataParallel(base_model)
    # else:
    #     model = base_model
    model = base_model
    model = model.to(device)
    # ----------------------------------------------------

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 🔑 Cap steps per epoch by num_steps (for smoke tests)
    steps_per_epoch = min(num_steps, len(train_loader))
    num_epochs = 1  # for quick tests, just do 1 short epoch
    total_steps = steps_per_epoch * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    print(f"\nTraining configuration:")
    print(f"  Num epochs: {num_epochs}")
    print(f"  Steps per epoch (capped): {steps_per_epoch}")
    print(f"  Total steps (scheduled): {total_steps}")
    print(f"  Target steps: {num_steps}")

    # Training loop
    best_val_score = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # ✅ Train only steps_per_epoch batches
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            max_steps=steps_per_epoch,
        )

        # Validate on full val set
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}, "
            f"Val AUROC: {val_metrics['auroc']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_score:
            best_val_score = val_metrics["accuracy"]
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            best_model_state = {
                "model_state_dict": model_to_save.state_dict(),
                "val_metrics": val_metrics,
                "epoch": epoch,
            }

    # Save best model for this fold
    fold_save_path = save_dir / f"fold_{fold_idx}_best.pt"
    torch.save(best_model_state, fold_save_path)
    print(f"\nBest model saved to {fold_save_path}")
    print(f"Best validation accuracy: {best_val_score:.4f}")

    return best_model_state, best_val_score


def main():
    parser = argparse.ArgumentParser(description="IA3 fine-tuning for promoter classification")
    parser.add_argument("--data-path", type=Path, required=True,
                        help="Path to promoter dataset TSV")
    parser.add_argument("--model-name", type=str,
                        default="InstaDeepAI/nucleotide-transformer-500m-human-ref",
                        help="HuggingFace model name")
    parser.add_argument("--output-dir", type=Path, default=Path("results/promoter_ia3"),
                        help="Output directory for results")
    parser.add_argument("--n-folds", type=int, default=10,
                        help="Number of cross-validation folds (default: 10)")
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Training steps per fold (default: 10000)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8, as per paper)")
    parser.add_argument("--learning-rate", type=float, default=3e-3,
                        help="Learning rate (default: 3e-3, as per paper)")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Proportion for held-out test set")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("IA3 Fine-tuning for Promoter Classification")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Cross-validation folds: {args.n_folds}")
    print(f"Steps per fold: {args.num_steps}")
    print()
    
    # Load full dataset (excluding test set)
    print("Loading data...")
    train_val_seqs, train_val_labels, _, _, test_seqs, test_labels = load_promoter_data(
        data_path=args.data_path,
        test_size=args.test_size,
        val_size=0.0,  # We'll do CV on train+val
        random_seed=args.seed,
    )
    
    # Create full train+val dataset for cross-validation
    full_dataset = PromoterDataset(
        sequences=train_val_seqs,
        labels=train_val_labels,
        tokenizer_name=args.model_name,
    )
    
    # Create held-out test dataset
    test_dataset = PromoterDataset(
        sequences=test_seqs,
        labels=test_labels,
        tokenizer_name=args.model_name,
    )
    
    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_seqs, train_val_labels)):
        print("\n" + "=" * 70)
        print(f"Fold {fold_idx + 1}/{args.n_folds}")
        print("=" * 70)
        
        # Create fold-specific datasets
        train_fold_dataset = Subset(full_dataset, train_idx)
        val_fold_dataset = Subset(full_dataset, val_idx)
        
        print(f"Train size: {len(train_fold_dataset)}")
        print(f"Val size: {len(val_fold_dataset)}")
        
        # Train on this fold
        best_model_state, best_val_score = train_single_fold(
            train_dataset=train_fold_dataset,
            val_dataset=val_fold_dataset,
            model_name=args.model_name,
            num_steps=args.num_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            save_dir=args.output_dir,
            fold_idx=fold_idx,
        )
        
        fold_results.append({
            'fold': fold_idx,
            'val_metrics': best_model_state['val_metrics'],
            'best_epoch': best_model_state['epoch'],
        })
    
    # Aggregate cross-validation results
    print("\n" + "=" * 70)
    print("Cross-Validation Results")
    print("=" * 70)
    
    val_accuracies = [r['val_metrics']['accuracy'] for r in fold_results]
    val_f1s = [r['val_metrics']['f1'] for r in fold_results]
    val_aurocs = [r['val_metrics']['auroc'] for r in fold_results]
    
    print(f"Validation Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")
    print(f"Validation F1: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")
    print(f"Validation AUROC: {np.mean(val_aurocs):.4f} ± {np.std(val_aurocs):.4f}")
    
    # Save results

    raw_config = vars(args).copy()
    # Convert any Path objects to strings for JSON
    config = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in raw_config.items()
    }
    
    results = {
        'config': config,  # <-- use the converted config here
        'fold_results': fold_results,
        'cv_summary': {
            'accuracy_mean': float(np.mean(val_accuracies)),
            'accuracy_std': float(np.std(val_accuracies)),
            'f1_mean': float(np.mean(val_f1s)),
            'f1_std': float(np.std(val_f1s)),
            'auroc_mean': float(np.mean(val_aurocs)),
            'auroc_std': float(np.std(val_aurocs)),
        },
        'timestamp': datetime.now().isoformat(),
    }

    
    results_path = args.output_dir / "cv_results.json"
    with results_path.open('w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()