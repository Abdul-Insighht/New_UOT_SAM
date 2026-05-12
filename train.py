"""
Enhanced_RRSIS_UOT Training Script

Train Enhanced RRSIS model with 4 novel techniques:
    1. Text-Guided Dynamic LoRA
    2. Contrastive Language-Image Loss
    3. Multi-Scale OT Feature Alignment
    4. OHEM + Focal + Boundary Loss

Supports: RRSIS-D, RRSIS-HR, RefSegRS datasets.

Usage:
    python train.py --dataset rrsis_d --data_root /path/to/data --sam3_ckpt ./pre-trained-weights/sam3.pt
"""

import os
import sys
import time
import random
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from args import get_args
from data.dataset import get_dataset, collate_fn
from lib.enhanced_model import Enhanced_RRSIS_UOT
from lib.rs_adapters import get_trainable_params_summary
import utils

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================
# Test-Time Augmentation (TTA)
# ============================================================
@torch.no_grad()
def tta_predict(model, images, captions, masks_gt=None):
    """
    Test-Time Augmentation: average predictions over flipped inputs.

    Averages logits from original, horizontally-flipped, and vertically-flipped
    images to produce more robust predictions. Free improvement, no retraining.

    Args:
        model: Enhanced_RRSIS_UOT model in eval mode.
        images: (B, 3, H, W) input images.
        captions: List[str] of length B.
        masks_gt: optional GT masks for loss computation.

    Returns:
        dict with 'pred_masks' (averaged logits) and optionally 'loss'.
    """
    preds = []

    # Original
    with torch.cuda.amp.autocast(enabled=True):
        out_orig = model(images, captions, masks_gt)
    preds.append(out_orig['pred_masks'])

    # Horizontal flip
    with torch.cuda.amp.autocast(enabled=True):
        out_hflip = model(torch.flip(images, [-1]), captions)
    preds.append(torch.flip(out_hflip['pred_masks'], [-1]))

    # Vertical flip
    with torch.cuda.amp.autocast(enabled=True):
        out_vflip = model(torch.flip(images, [-2]), captions)
    preds.append(torch.flip(out_vflip['pred_masks'], [-2]))

    # Average logits
    avg_mask = torch.stack(preds).mean(dim=0)

    result = {'pred_masks': avg_mask}
    if 'loss' in out_orig:
        result['loss'] = out_orig['loss']
    return result


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Tracks average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_iou(pred, target, threshold=0.5, return_components=False):
    """Compute IoU between predicted and target masks."""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    if return_components:
        return iou, intersection.sum().item(), union.sum().item()
    return iou.mean().item()


def get_optimizer(model, args):
    """
    Create optimizer with parameter groups for differentiated learning rates.

    Groups:
        1. LoRA / Dynamic LoRA parameters — lr_backbone
        2. Enhancement modules (OT, contrastive, OHEM) — lr_decoder
        3. Decoder/encoder parameters — lr_decoder
        4. Other trainable params — lr
    """
    lora_params = []
    enhancement_params = []
    decoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora' in name.lower() or 'dynamic_lora' in name.lower() or 'hyper_' in name.lower():
            lora_params.append(param)
        elif any(x in name for x in ['ms_ot_aligner', 'contrastive_loss', 'enhanced_loss',
                                      'ot_aligner', 'mask_refine_head', 'orient']):
            enhancement_params.append(param)
        elif any(x in name for x in ['transformer', 'segmentation_head', 'geometry_encoder', 'dot_prod']):
            decoder_params.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {'params': lora_params, 'lr': args.lr_backbone, 'name': 'lora_adapters'},
        {'params': enhancement_params, 'lr': args.lr_decoder, 'name': 'enhancements'},
        {'params': decoder_params, 'lr': args.lr_decoder, 'name': 'decoder'},
        {'params': other_params, 'lr': args.lr, 'name': 'other'},
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    for g in param_groups:
        print(f"  Param group '{g['name']}': {sum(p.numel() for p in g['params']):,} params, lr={g['lr']}")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return optimizer


def get_scheduler(optimizer, args, steps_per_epoch):
    """Create learning rate scheduler with warmup."""
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


@torch.no_grad()
def validate(model, val_loader, device, epoch, use_tta=False):
    """Run validation and compute metrics."""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: '
    
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    total_loss = 0
    total_its = 0

    for images, masks, captions in metric_logger.log_every(val_loader, 100, header):
        total_its += 1
        images = images.to(device)
        masks = masks.to(device)

        # Handle eval mode captions (list of lists → flatten)
        if isinstance(captions[0], list):
            captions = [cap[0] for cap in captions]

        # Use TTA if enabled
        if use_tta:
            outputs = tta_predict(model, images, captions, masks)
        else:
            with torch.cuda.amp.autocast(enabled=True):
                outputs = model(images, captions, masks)

        loss = outputs['loss'].item()
        total_loss += loss

        iou_tensor, I, U = compute_iou(outputs['pred_masks'], masks, return_components=True)
        
        for iou in iou_tensor:
            iou_val = iou.item()
            mean_IoU.append(iou_val)
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou_val >= eval_seg_iou)
            seg_total += 1

        cum_I += I
        cum_U += U

    mIoU = np.mean(mean_IoU)
    print('Final results: Mean IoU is %.2f%%' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    Precision@%s = %.2f%%\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    overall_iou = (cum_I * 100. / cum_U) if cum_U > 0 else 0
    results_str += '    Overall IoU = %.2f%%\n' % overall_iou
    print(results_str)

    avg_loss = total_loss / max(total_its, 1)
    return mIoU * 100., overall_iou


def train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('weight_decay', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('iou', utils.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)

    optimizer.zero_grad()

    for batch_idx, (images, masks, captions) in enumerate(metric_logger.log_every(train_loader, args.print_freq, header)):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(images, captions, masks)
            loss = outputs['loss'] / args.grad_accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        # Metrics
        with torch.no_grad():
            iou = compute_iou(outputs['pred_masks'], masks)

        # Log individual loss components
        log_dict = {
            'loss': outputs['loss'].item(),
            'iou': iou,
            'lr': optimizer.param_groups[0]["lr"],
            'weight_decay': optimizer.param_groups[0].get("weight_decay", 0.0),
        }
        if 'seg_loss' in outputs:
            log_dict['seg_loss'] = outputs['seg_loss'].item() if isinstance(outputs['seg_loss'], torch.Tensor) else outputs['seg_loss']
        if 'contrastive_loss' in outputs:
            log_dict['cl_loss'] = outputs['contrastive_loss'].item() if isinstance(outputs['contrastive_loss'], torch.Tensor) else outputs['contrastive_loss']

        metric_logger.update(**log_dict)

    return metric_logger.meters['loss'].global_avg, metric_logger.meters['iou'].global_avg


def main():
    args = get_args()
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  Enhanced_RRSIS_UOT Training")
    print(f"  Dataset: {args.dataset}")
    print(f"  Image Size: {args.image_size}×{args.image_size}")
    print(f"  LoRA Rank: {args.lora_rank}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size} × {args.grad_accum_steps} accum")
    print(f"  Learning Rate: {args.lr} (Backbone: {args.lr_backbone}, Decoder: {args.lr_decoder})")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Device: {device}")
    print(f"  --- Enhancements ---")
    print(f"  Dynamic LoRA: {args.use_dynamic_lora}")
    print(f"  Contrastive Loss: {args.use_contrastive_loss} (weight={args.contrastive_weight})")
    print(f"  Multi-Scale OT: {args.use_multiscale_ot} ({args.num_ot_scales} scales)")
    print(f"  UOT Alpha: {args.ot_alpha}, Beta: {args.ot_beta}, Gamma(reg): {args.ot_reg}")
    print(f"  UOT Learnable Margins: {args.learnable_margins} (warmup={args.uot_warmup_epochs} epochs)")
    print(f"  OHEM Loss: {args.use_ohem_loss} (hard_ratio={args.ohem_hard_ratio})")

    print(f"  Data Augmentation: {getattr(args, 'use_augmentation', True)}")
    print(f"  TTA: {getattr(args, 'use_tta', True)}")
    print(f"{'='*60}\n")

    # ====== Build Enhanced Model ======
    print("Building Enhanced_RRSIS_UOT model...")
    model = Enhanced_RRSIS_UOT(
        sam3_ckpt=args.sam3_ckpt,
        image_size=args.image_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        freeze_backbone=args.freeze_backbone,
        freeze_text_encoder=args.freeze_text_encoder,
        gradient_checkpointing=args.gradient_checkpointing,
        # Enhancement flags
        use_dynamic_lora=args.use_dynamic_lora,
        use_contrastive_loss=args.use_contrastive_loss,
        use_multiscale_ot=args.use_multiscale_ot,
        use_ohem_loss=args.use_ohem_loss,

        # Enhancement params
        contrastive_weight=args.contrastive_weight,
        ohem_hard_ratio=args.ohem_hard_ratio,
        ot_reg=args.ot_reg,
        ot_alpha=args.ot_alpha,
        ot_beta=args.ot_beta,
        ot_num_iter=args.ot_num_iter,
        num_ot_scales=args.num_ot_scales,
        learnable_margins=args.learnable_margins,
        uot_warmup_epochs=args.uot_warmup_epochs,

    )
    model = model.to(device)

    # ====== Build Datasets ======
    print("\nLoading datasets...")
    train_dataset = get_dataset(args, split='train', eval_mode=False)
    val_dataset = get_dataset(args, split='val', eval_mode=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ====== Optimizer & Scheduler ======
    optimizer = get_optimizer(model, args)
    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    scheduler = get_scheduler(optimizer, args, steps_per_epoch)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # ====== Resume ======
    start_epoch = 0
    best_iou = 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_iou = ckpt.get('best_iou', 0.0)

        # Restore scheduler state if available
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            print(f"  Scheduler state restored.")
        else:
            # Fast-forward scheduler to the correct step
            for _ in range(start_epoch * steps_per_epoch):
                scheduler.step()
            print(f"  Scheduler fast-forwarded to step {start_epoch * steps_per_epoch}.")

        # Restore scaler state if available
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            print(f"  GradScaler state restored.")

        # Override optimizer LR with new args (for fine-tuning with lower LR)
        lr_map = {
            'lora_adapters': args.lr_backbone,
            'enhancements': args.lr_decoder,
            'decoder': args.lr_decoder,
            'other': args.lr,
        }
        for i, pg in enumerate(optimizer.param_groups):
            pg_name = pg.get('name', 'other')
            new_lr = lr_map.get(pg_name, args.lr)
            print(f"  Override LR for '{pg_name}': {pg['lr']:.2e} → {new_lr:.2e}")
            pg['lr'] = new_lr
            pg['initial_lr'] = new_lr  # Important: Update initial_lr for scheduler
            pg['weight_decay'] = args.weight_decay

            
            # Also update the scheduler's base_lrs if it exists
            if hasattr(scheduler, 'base_lrs') and i < len(scheduler.base_lrs):
                scheduler.base_lrs[i] = new_lr

        print(f"  Resumed at epoch {start_epoch}, best_iou={best_iou:.4f}")

    # ====== Output Directory ======
    os.makedirs(args.output_dir, exist_ok=True)

    # ====== Wandb ======
    if HAS_WANDB:
        wandb.init(
            project="Enhanced_RRSIS_UOT",
            name=f"{args.dataset}_enhanced_lr{args.lr}_lora{args.lora_rank}",
            config=vars(args),
        )

    # ====== Training Loop ======
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        # Update UOT warmup state (unfreeze alpha/beta after warmup)
        model.set_epoch(epoch)

        # Log current UOT alpha/beta values
        uot_info = model.get_uot_info()
        if uot_info:
            uot_str = ', '.join(f'{k}={v:.4f}' for k, v in uot_info.items())
            print(f"  [UOT] {uot_str}")

        # Train
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch + 1, args
        )

        # Validate
        val_iou, val_overall_iou = validate(
            model, val_loader, device, epoch + 1,
            use_tta=getattr(args, 'use_tta', False),
        )
        
        print('Average object IoU {}'.format(val_iou))
        print('Overall IoU {}'.format(val_overall_iou))

        # Wandb logging
        if HAS_WANDB:
            log_dict = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_iou': val_iou,
                'val_overall_iou': val_overall_iou,
                'lr': optimizer.param_groups[0]['lr'],
            }
            # Log UOT alpha/beta values
            uot_info = model.get_uot_info()
            if uot_info:
                log_dict.update({f'uot/{k}': v for k, v in uot_info.items()})
            wandb.log(log_dict)

        # Save best model
        is_best = (val_iou + val_overall_iou) > best_iou
        if is_best:
            best_iou = val_iou + val_overall_iou
            print('Better epoch: {}\n'.format(epoch))
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_iou': best_iou,
                'val_iou': val_iou,
                'val_overall_iou': val_overall_iou,
                'args': vars(args),
            }, save_path)
            print(f"  ★ New best model saved! mIoU={val_iou:.2f}, oIoU={val_overall_iou:.2f}")

        # Save latest model at every epoch
        latest_save_path = os.path.join(args.output_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_iou': best_iou,
            'val_iou': val_iou,
            'val_overall_iou': val_overall_iou,
            'args': vars(args),
        }, latest_save_path)
        print(f"  Latest model weights updated for epoch {epoch+1}.")

        # Save checkpoint every 5 epochs (more frequent for monitoring)
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_iou': best_iou,
                'val_iou': val_iou,
                'val_overall_iou': val_overall_iou,
                'args': vars(args),
            }, save_path)
            print(f"  Checkpoint saved: {save_path}")

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best mIoU: {best_iou:.4f}")
    print(f"{'='*60}")

    if HAS_WANDB:
        wandb.finish()


if __name__ == '__main__':
    main()
