"""
Training Script for Indian Butterfly Classification
====================================================
Features:
  - Auto-selects GPU with most free memory
  - Mixed precision (AMP) for V100 efficiency
  - TensorBoard logging & Progress bar via tqdm
  - Custom I/O logging to progress.log
  - Best checkpoint saving by val macro-F1
  - Cosine annealing LR schedule with warmup
  - Class-balanced sampling
  - Precision & Accuracy tracking with CSV logging
"""

import argparse
import os
import sys
import time
import json
import csv
import logging
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score, precision_score
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from dataset import create_dataloaders
    from models.backbone import build_model
    from losses import build_loss
except ImportError:
    pass  # Allow IDE resolution if running from different dir


def setup_logger(log_file):
    """Setup custom text logger appended dynamically."""
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def select_gpu() -> int:
    """Auto-select GPU with most free memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,index',
             '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        gpus = []
        for line in lines:
            parts = line.strip().split(',')
            free_mb = int(parts[0].strip())
            idx = int(parts[1].strip())
            gpus.append((free_mb, idx))
        # Pick GPU with most free memory
        gpus.sort(reverse=True)
        best_free, best_idx = gpus[0]
        print(f"  Auto-selected GPU {best_idx} ({best_free} MB free)")
        return best_idx
    except Exception as e:
        print(f"  GPU auto-select failed, using GPU 0")
        return 0


def plot_training_curves(csv_path, out_dir):
    """Reads metrics.csv and outputs visual accuracy/precision/loss graphs."""
    epochs, t_loss, v_loss, t_acc, v_acc, v_f1, v_prec = [], [], [], [], [], [], []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            t_loss.append(float(row['train_loss']))
            v_loss.append(float(row['val_loss']))
            t_acc.append(float(row['train_acc']))
            v_acc.append(float(row['val_acc']))
            v_f1.append(float(row['val_f1']))
            v_prec.append(float(row['val_precision']))

    if not epochs:
        return

    # Accuracy / F1 / Precision
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, t_acc, label='Train Acc', marker='o')
    plt.plot(epochs, v_acc, label='Val Acc', marker='s')
    plt.plot(epochs, v_f1, label='Val Macro-F1', marker='^', linestyle='--')
    plt.plot(epochs, v_prec, label='Val Precision', marker='d', linestyle=':')
    plt.title('Training & Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'metrics_curve.png'))
    plt.close()

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, t_loss, label='Train Loss', color='red', marker='o')
    plt.plot(epochs, v_loss, label='Val Loss', color='darkorange', marker='s')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, use_geo, epoch, logger
):
    model.train()
    total_loss = 0
    ema_loss = None
    all_preds = []
    all_targets = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} Train", dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        zone_idx = batch.get('zone_idx')
        month_enc = batch.get('month_enc')
        if use_geo and zone_idx is not None:
            zone_idx = zone_idx.to(device, non_blocking=True)
            month_enc = month_enc.to(device, non_blocking=True)
        else:
            zone_idx = None
            month_enc = None

        optimizer.zero_grad()

        with autocast(device_type='cuda' if torch.cuda.is_available() and device.type == 'cuda' else 'cpu'):
            logits = model(images, zone_idx, month_enc)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        curr_loss = loss.item()
        total_loss += curr_loss
        
        # Calculate EMA loss for smooth printing
        if ema_loss is None:
            ema_loss = curr_loss
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * curr_loss

        preds = logits.argmax(dim=1).cpu().numpy()
        targets_np = labels.cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets_np)
        
        # Evaluate total species currently evaluated dynamically
        eval_species_count = len(set(targets_np))

        pbar.set_postfix({'loss': f"{curr_loss:.4f}", 'ema_loss': f"{ema_loss:.4f}"})
        
        if batch_idx % 20 == 0 or batch_idx == len(loader) - 1:
            logger.info(
                f"[Epoch {epoch}][Batch {batch_idx}/{len(loader)}] "
                f"Loss: {curr_loss:.4f} | EMA Loss: {ema_loss:.4f} | "
                f"Batch Species Covered: {eval_species_count} | Total Evaluated: {len(all_targets)}"
            )

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, macro_f1


@torch.no_grad()
def validate(model, loader, criterion, device, use_geo, epoch, logger):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} Val  ", dynamic_ncols=True, leave=False)
    for batch in pbar:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        zone_idx = batch.get('zone_idx')
        month_enc = batch.get('month_enc')
        if use_geo and zone_idx is not None:
            zone_idx = zone_idx.to(device, non_blocking=True)
            month_enc = month_enc.to(device, non_blocking=True)
        else:
            zone_idx = None
            month_enc = None

        with autocast(device_type='cuda' if torch.cuda.is_available() and device.type == 'cuda' else 'cpu'):
            logits = model(images, zone_idx, month_enc)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(loader), 1)
    acc = accuracy_score(all_targets, all_preds)
    # Generate precision and F1
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    logger.info(
        f"[Epoch {epoch} VAL] Avg Loss: {avg_loss:.4f} | Acc: {acc:.4f} | "
        f"Precision: {precision:.4f} | Macro F1: {macro_f1:.4f} | "
        f"Total Samples: {len(all_targets)}"
    )

    return avg_loss, acc, macro_f1, precision


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description='Butterfly Classification Training')

    # parser.add_argument('--data_root', type=str, default='/data/butterflies')
    parser.add_argument('--data_root', type=str, default='/data/butterflies')
    parser.add_argument('--metadata_file', type=str, default='metadata_filtered.csv')
    parser.add_argument('--img_size', type=int, default=224)

    # Phase defaults to 3 (MLFI + CA) to balance robust learning against extreme scattering.
    parser.add_argument('--phase', type=int, default=3, choices=[1, 2, 3, 5])
    parser.add_argument('--use_geotemporal', action='store_true')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--loss', type=str, default='focal', choices=['ce', 'focal'])
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--balanced_sampling', type=bool, default=True)

    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default='./runs')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--resume', type=str, default='')

    args = parser.parse_args()

    # Create run directory
    if not args.exp_name:
        phase_names = {1: 'baseline', 2: 'ca', 3: 'mlfi', 5: 'geo'}
        args.exp_name = (
            f"phase{args.phase}_{phase_names[args.phase]}_"
            f"{args.loss}_bs{args.batch_size}_lr{args.lr}"
        )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f"{args.exp_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Setup core logger explicitly to file to isolate tracking stats later
    logger = setup_logger(os.path.join(run_dir, 'progress.log'))
    
    metrics_csv_path = os.path.join(run_dir, 'metrics.csv')
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'val_f1', 'val_precision', 'lr'])

    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'#'*60}")
    print(f"  BUTTERFLY CLASSIFICATION TRAINING")
    print(f"  Experiment: {args.exp_name}")
    print(f"  Output: {run_dir}")
    print(f"{'#'*60}\n")
    logger.info("Starting Run Initialization...")

    if args.gpu == -1:
        gpu_idx = select_gpu()
    else:
        gpu_idx = args.gpu
    device = torch.device(f'cuda:{gpu_idx}' if torch.cuda.is_available() else 'cpu')
    
    use_geo = args.use_geotemporal or args.phase == 5
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        use_geotemporal=use_geo,
        balanced_sampling=args.balanced_sampling,
        metadata_file=args.metadata_file,
    )

    model = build_model(num_classes=num_classes, phase=args.phase, pretrained=args.pretrained, dropout=args.dropout)
    model = model.to(device)

    class_weights = train_loader.dataset.get_class_weights().to(device) if args.loss == 'focal' else None
    criterion = build_loss(
        loss_type=args.loss,
        class_weights=class_weights,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    backbone_params = list(model.backbone.parameters())
    new_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1}, 
        {'params': new_params, 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)
    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=os.path.join(run_dir, 'tb_logs'))

    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_f1 = ckpt.get('best_f1', 0.0)

    print(f"\n  Starting training: {args.epochs} epochs")
    epochs_pbar = tqdm(range(start_epoch, args.epochs), desc="Training Epochs", total=args.epochs - start_epoch)
    for epoch in epochs_pbar:
        
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_geo, epoch, logger
        )

        val_loss, val_acc, val_f1, val_prec = validate(
            model, val_loader, criterion, device, use_geo, epoch, logger
        )

        scheduler.step()
        current_lr = optimizer.param_groups[-1]['lr']

        # Log to metric csv
        with open(metrics_csv_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{train_acc:.4f}", f"{val_acc:.4f}", f"{val_f1:.4f}", f"{val_prec:.4f}", f"{current_lr:.6f}"])

        # Plot dynamic graph end-of-epoch
        plot_training_curves(metrics_csv_path, run_dir)

        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('accuracy', {'train': train_acc, 'val': val_acc}, epoch)
        writer.add_scalar('lr', current_lr, epoch)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'best_f1': best_f1,
                'config': vars(args),
            }, os.path.join(run_dir, 'best_model.pth'))
            logger.info(f"*** New best model saved! Val Macro-F1: {val_f1:.4f} ***")

        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_f1': best_f1, 'config': vars(args)}, os.path.join(run_dir, f'checkpoint_epoch{epoch+1}.pth'))
        
        epochs_pbar.set_postfix({'val_f1': f"{val_f1:.4f}", 'best_f1': f"{best_f1:.4f}", 'lr': f"{current_lr:.6f}"})

    epochs_pbar.close()
    logger.info("Training Run Complete! Check metrics graphs in the output directory.")
    writer.close()

if __name__ == '__main__':
    main()
