"""
OHEM (Online Hard Example Mining) + Focal + Boundary-Aware Loss
for Enhanced_RRSIS_UOT.

Remote sensing images have extreme foreground-background imbalance
(often 98%+ background). These loss functions focus training on
the hardest, most informative pixels.

Three components:
    1. OHEMLoss: Selects top-K hardest pixels for loss computation
    2. FocalDiceLoss: Focal weighting + Dice for class imbalance
    3. BoundaryAwareLoss: Extra supervision on object boundaries

Reference:
    Shrivastava et al., "Training Region-based Object Detectors with
    Online Hard Example Mining", CVPR 2016.
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining loss.

    Instead of computing loss over ALL pixels, selects the top-K hardest
    (highest loss) pixels and computes loss only on them. This focuses
    the model on difficult boundary regions and misclassified areas.

    Args:
        hard_ratio: Fraction of hardest pixels to keep (0.0-1.0).
        min_kept: Minimum number of pixels to keep per sample.
    """

    def __init__(self, hard_ratio=0.3, min_kept=256):
        super().__init__()
        self.hard_ratio = hard_ratio
        self.min_kept = min_kept

    def forward(self, pred_logits, gt_masks):
        """
        Compute OHEM-weighted BCE loss.

        Args:
            pred_logits: (B, 1, H, W) predicted mask logits.
            gt_masks: (B, 1, H, W) ground truth binary masks.

        Returns:
            ohem_loss: scalar tensor.
        """
        # Resize GT if needed
        if pred_logits.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.float(), pred_logits.shape[-2:], mode='nearest'
            )

        B = pred_logits.shape[0]

        # Compute per-pixel BCE loss (no reduction)
        pixel_loss = F.binary_cross_entropy_with_logits(
            pred_logits, gt_masks.float(), reduction='none'
        )  # (B, 1, H, W)

        total_loss = 0.0
        for i in range(B):
            sample_loss = pixel_loss[i].flatten()  # (H*W,)
            num_pixels = sample_loss.numel()

            # Number of hard pixels to keep
            num_hard = max(int(num_pixels * self.hard_ratio), self.min_kept)
            num_hard = min(num_hard, num_pixels)

            # Select top-K hardest pixels
            topk_loss, _ = torch.topk(sample_loss, num_hard)
            total_loss += topk_loss.mean()

        return total_loss / B


class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss for handling class imbalance.

    Focal loss down-weights easy examples (well-classified background pixels)
    and up-weights hard examples (boundary/misclassified pixels).
    Dice loss handles global overlap.

    Args:
        alpha: Focal loss balancing factor for positive class.
        gamma: Focal loss focusing parameter (higher = more focus on hard).
        dice_weight: Weight for Dice loss component.
        focal_weight: Weight for Focal loss component.
    """

    def __init__(self, alpha=0.75, gamma=2.0, dice_weight=5.0, focal_weight=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred_logits, gt_masks):
        """
        Compute Focal + Dice loss.

        Args:
            pred_logits: (B, 1, H, W) predicted mask logits.
            gt_masks: (B, 1, H, W) ground truth binary masks.

        Returns:
            combined_loss: scalar tensor.
        """
        # Resize GT if needed
        if pred_logits.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.float(), pred_logits.shape[-2:], mode='nearest'
            )
        gt = gt_masks.float()

        # === Focal Loss ===
        pred_prob = torch.sigmoid(pred_logits)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, gt, reduction='none'
        )

        # Focal weighting: (1 - p_t)^gamma
        p_t = pred_prob * gt + (1 - pred_prob) * (1 - gt)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * gt + (1 - self.alpha) * (1 - gt)

        focal_loss = (alpha_t * focal_weight * bce).mean()

        # === Dice Loss ===
        pred_prob_clamped = pred_prob.clamp(min=1e-6, max=1.0 - 1e-6)
        intersection = (pred_prob_clamped * gt).sum(dim=(2, 3))
        union = pred_prob_clamped.sum(dim=(2, 3)) + gt.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice.mean()

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


class EnhancedOHEMLoss(nn.Module):
    """
    Combined loss with OHEM components.

    Aggregates OHEM and FocalDice losses with
    configurable weights.

    Args:
        ohem_weight: Weight for OHEM hard pixel loss.
        focal_dice_weight: Weight for Focal + Dice loss.
        hard_ratio: OHEM hard pixel ratio.
        focal_gamma: Focal loss gamma.
        score_weight: Weight for query confidence loss.
    """

    def __init__(
        self,
        ohem_weight=1.0,
        focal_dice_weight=1.0,
        hard_ratio=0.3,
        focal_gamma=2.0,
        score_weight=1.0,
    ):
        super().__init__()
        self.ohem_weight = ohem_weight
        self.focal_dice_weight = focal_dice_weight
        self.score_weight = score_weight

        self.ohem_loss = OHEMLoss(hard_ratio=hard_ratio)
        self.focal_dice_loss = FocalDiceLoss(gamma=focal_gamma)

        print("[EnhancedOHEM] OHEM + FocalDice loss initialized")

    def forward(self, outputs, gt_masks, image_size):
        """
        Compute combined OHEM + FocalDice loss.

        Args:
            outputs: dict with 'pred_masks' and optionally 'pred_logits'.
            gt_masks: (B, 1, H, W) ground truth masks.
            image_size: int, spatial size.

        Returns:
            total_loss: scalar tensor.
        """
        pred_masks = outputs['pred_masks']

        # 1. OHEM loss (hard pixel mining)
        ohem = self.ohem_loss(pred_masks, gt_masks)

        # 2. Focal + Dice loss (class imbalance handling)
        focal_dice = self.focal_dice_loss(pred_masks, gt_masks)

        # 3. Score supervision (optional)
        score_loss = torch.tensor(0.0, device=gt_masks.device)
        if 'pred_logits' in outputs and outputs['pred_logits'] is not None:
            score_loss = self._compute_score_loss(outputs['pred_logits'], gt_masks)

        total = (
            self.ohem_weight * ohem +
            self.focal_dice_weight * focal_dice +
            self.score_weight * score_loss
        )

        return total

    def _compute_score_loss(self, pred_logits, gt_masks):
        """Supervise query confidence scores."""
        B = gt_masks.shape[0]
        device = gt_masks.device

        if pred_logits.dim() == 3:
            scores = pred_logits.squeeze(-1)
            N = scores.shape[1]
            with torch.no_grad():
                has_object = (gt_masks.sum(dim=(1, 2, 3)) > 0).float()
                target_scores = has_object.unsqueeze(1).expand(B, N) / N
            loss = F.binary_cross_entropy_with_logits(
                scores, target_scores, reduction='mean'
            )
            return loss

        return torch.tensor(0.0, device=device)
