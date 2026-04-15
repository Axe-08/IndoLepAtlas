"""
Evaluation Script for Indian Butterfly Classification
=====================================================
Outputs:
  - Top-1 and Top-5 accuracy
  - Precision, Macro-averaged F1 score, Weighted F1
  - Per-class accuracy breakdown
  - Per-stratum accuracy (dense vs sparse classes)
  - Confusion matrix (top-20 most confused pairs)
  - Live progress tracking with tqdm
"""

import argparse
import os
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, classification_report,
    confusion_matrix
)
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from dataset import ButterflyDataset, get_val_transforms
    from models.backbone import build_model
except ImportError:
    pass


@torch.no_grad()
def run_inference(model, loader, device, use_geo=False):
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []

    pbar = tqdm(loader, desc="Evaluation Inference", dynamic_ncols=True)
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label']

        zone_idx = batch.get('zone_idx')
        month_enc = batch.get('month_enc')
        if use_geo and zone_idx is not None:
            zone_idx = zone_idx.to(device, non_blocking=True)
            month_enc = month_enc.to(device, non_blocking=True)
        else:
            zone_idx = None
            month_enc = None

        with autocast():
            logits = model(images, zone_idx, month_enc)

        all_logits.append(logits.cpu())
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_targets.extend(labels.numpy())
        
        pbar.set_postfix({'completed_samples': len(all_targets)})

    return np.array(all_preds), np.array(all_targets), torch.cat(all_logits, dim=0)


def compute_topk_accuracy(logits, targets, k=5):
    topk_preds = logits.topk(k, dim=1).indices.numpy()
    correct = sum(1 for t, preds in zip(targets, topk_preds) if t in preds)
    return correct / len(targets)


def analyze_confusion(preds, targets, idx_to_species, top_n=20):
    cm = confusion_matrix(targets, preds)
    n = cm.shape[0]

    confusions = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], idx_to_species[i], idx_to_species[j]))

    confusions.sort(reverse=True)
    return confusions[:top_n]


def per_stratum_accuracy(preds, targets, class_counts, idx_to_species, threshold=50):
    dense_classes = set(i for i, c in enumerate(class_counts) if c >= threshold)
    sparse_classes = set(i for i, c in enumerate(class_counts) if c < threshold)

    dense_mask = np.array([t in dense_classes for t in targets])
    sparse_mask = np.array([t in sparse_classes for t in targets])

    results = {}
    if dense_mask.any():
        results['dense'] = {
            'n_classes': len(dense_classes),
            'n_samples': int(dense_mask.sum()),
            'accuracy': float(accuracy_score(targets[dense_mask], preds[dense_mask])),
            'macro_f1': float(f1_score(targets[dense_mask], preds[dense_mask],
                                  average='macro', zero_division=0)),
        }
    if sparse_mask.any():
        results['sparse'] = {
            'n_classes': len(sparse_classes),
            'n_samples': int(sparse_mask.sum()),
            'accuracy': float(accuracy_score(targets[sparse_mask], preds[sparse_mask])),
            'macro_f1': float(f1_score(targets[sparse_mask], preds[sparse_mask],
                                  average='macro', zero_division=0)),
        }
    return results


def plot_confusion_heatmap(preds, targets, idx_to_species, out_path, top_n=30):
    cm = confusion_matrix(targets, preds)
    errors_per_class = cm.sum(axis=1) - np.diag(cm)
    top_error_classes = np.argsort(errors_per_class)[-top_n:]

    sub_cm = cm[np.ix_(top_error_classes, top_error_classes)]
    labels = [idx_to_species.get(i, f'cls_{i}') for i in top_error_classes]
    short_labels = [l.split(' ')[1][:15] if ' ' in l else l[:15] for l in labels]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(sub_cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=short_labels, yticklabels=short_labels, ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix (Top-{top_n} Most Confused Species)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Butterfly Classification Evaluation')
    parser.add_argument('--data_root', type=str, default='/data/butterflies')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    parser.add_argument('--sparse_threshold', type=int, default=30)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"\n  Loading checkpoint: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
    except FileNotFoundError:
        print(f"Error: Checkpoint {args.checkpoint} not found. Please train the model first.")
        return
        
    config = ckpt.get('config', {})
    phase = config.get('phase', 1)
    use_geo = config.get('use_geotemporal', False) or phase == 5
    img_size = config.get('img_size', 224)

    dataset = ButterflyDataset(
        data_root=args.data_root,
        split=args.split,
        img_size=img_size,
        use_geotemporal=use_geo,
        metadata_file=config.get('metadata_file', 'metadata_filtered.csv'),
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    num_classes = dataset.num_classes

    model = build_model(
        num_classes=num_classes,
        phase=phase,
        pretrained=False,
        dropout=config.get('dropout', 0.3),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    print(f"\n  Running inference on {args.split} set ({len(dataset)} images)...")
    preds, targets, logits = run_inference(model, loader, device, use_geo)

    top1_acc = accuracy_score(targets, preds)
    top5_acc = compute_topk_accuracy(logits, targets, k=5)
    
    macro_precision = precision_score(targets, preds, average='macro', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS ({args.split.upper()})")
    print(f"{'='*60}")
    print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"  Macro Precision:{macro_precision:.4f}")
    print(f"  Macro F1:       {macro_f1:.4f}")
    print(f"  Weighted F1:    {weighted_f1:.4f}")

    # Generate stratum results to observe long-tail scaling
    train_ds = ButterflyDataset(args.data_root, 'train', use_geotemporal=use_geo)
    train_counts = train_ds.get_class_counts()

    stratum_results = per_stratum_accuracy(
        preds, targets, train_counts, dataset.idx_to_species, args.sparse_threshold
    )
    print(f"\n  Per-Stratum Analysis (threshold={args.sparse_threshold} images in dense constraint):")
    for stratum, stats in stratum_results.items():
        print(f"    {stratum.upper()}: {stats['n_classes']} classes, "
              f"{stats['n_samples']} samples, "
              f"acc={stats['accuracy']:.4f}, F1={stats['macro_f1']:.4f}")

    confusions = analyze_confusion(preds, targets, dataset.idx_to_species, top_n=20)
    print(f"\n  Top-20 Most Confused Pairs:")
    print(f"  {'Count':<8} {'True Species':<30} {'Predicted As':<30}")
    print(f"  {'-'*68}")
    for cnt, true_sp, pred_sp in confusions:
        print(f"  {cnt:<8} {true_sp:<30} {pred_sp:<30}")

    plot_confusion_heatmap(
        preds, targets, dataset.idx_to_species,
        os.path.join(args.output_dir, f'confusion_{args.split}.png')
    )

    results = {
        'split': args.split,
        'checkpoint': args.checkpoint,
        'phase': phase,
        'top1_accuracy': float(top1_acc),
        'top5_accuracy': float(top5_acc),
        'macro_precision': float(macro_precision),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'stratum_results': stratum_results,
        'top_confusions': [
            {'count': int(c), 'true': t, 'predicted': p}
            for c, t, p in confusions
        ],
    }
    out_file = os.path.join(args.output_dir, f'eval_{args.split}.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_file}")


if __name__ == '__main__':
    main()
