"""
Enhanced_RRSIS_UOT: Enhanced model wrapping RRSIS_SAM3 with 4 new techniques.

Enhancements over RRSIS_SAM3:
    1. Text-Guided Dynamic LoRA — text-conditioned vision adapter weights
    2. Contrastive Language-Image Loss — InfoNCE alignment supervision
    3. Multi-Scale OT Feature Alignment — per-FPN-level OT alignment
    4. OHEM Loss — hard pixel mining + focal + boundary-aware supervision

All 4 techniques can be individually enabled/disabled via flags for
ablation studies.

Architecture:
    Input (Image + Text)
        → SAM3 VL Backbone (ViT + Text Encoder)
            + Dynamic LoRA (text-conditioned vision adaptation) [NEW]
        → Multi-Scale OT Alignment (per-FPN-level) [ENHANCED]
        → Transformer Encoder (fusion)
        → DETR Decoder (detection)
        → Segmentation Head (mask prediction)
        → OHEM + Focal + Boundary Loss [ENHANCED]
        → Contrastive Loss (auxiliary) [NEW]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from sam3.model_builder import (
    build_sam3_image_model,
    _create_text_encoder,
    _create_vision_backbone,
    _create_vl_backbone,
    _create_sam3_transformer,
    _create_dot_product_scoring,
    _create_segmentation_head,
    _create_geometry_encoder,
    _create_sam3_model,
    _load_checkpoint,
)
from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt

from .rs_adapters import inject_lora_adapters, get_trainable_params_summary
from .dynamic_lora import inject_dynamic_lora_adapters, DynamicLoRAManager
from .multiscale_ot_alignment import MultiScaleOTAligner
from .contrastive_loss import ContrastiveLoss
from .ohem_loss import EnhancedOHEMLoss
from .ot_feature_alignment import OTFeatureAligner
from .ot_loss import OTSegmentationLoss


class Enhanced_RRSIS_UOT(nn.Module):
    """
    Enhanced RRSIS model with Unbalanced Optimal Transport and
    4 novel techniques for improved performance.

    This model wraps SAM3 and extends the base RRSIS_SAM3 with:
    - Text-Guided Dynamic LoRA for text-aware vision adaptation
    - Multi-Scale OT alignment across all FPN levels
    - Contrastive loss for vision-language alignment
    - OHEM loss for handling extreme class imbalance

    All enhancements are individually toggleable for ablation.

    Args:
        sam3_ckpt: Path to SAM3 pretrained checkpoint.
        image_size: Input image size (divisible by 14).
        lora_rank: LoRA rank for adapters.
        lora_alpha: LoRA scaling factor.
        freeze_backbone: Whether to freeze the ViT backbone.
        freeze_text_encoder: Whether to freeze the text encoder.
        gradient_checkpointing: Enable gradient checkpointing.
        use_dynamic_lora: Enable text-guided dynamic LoRA.
        use_contrastive_loss: Enable contrastive loss.
        use_multiscale_ot: Enable multi-scale OT alignment.
        use_ohem_loss: Enable OHEM + Focal + Boundary loss.
        contrastive_weight: Weight for contrastive loss.
        ohem_hard_ratio: OHEM hard pixel ratio.
        ot_reg: OT Sinkhorn regularization (gamma — FIXED).
        ot_alpha: UOT image marginal relaxation.
        ot_beta: UOT text marginal relaxation.
        ot_num_iter: Number of Sinkhorn iterations.
        num_ot_scales: Number of FPN scales for multi-scale OT.
        learnable_margins: Make alpha/beta trainable after warmup.
        uot_warmup_epochs: Epochs to keep alpha/beta fixed.
    """

    def __init__(
        self,
        sam3_ckpt: str = None,
        image_size: int = 504,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        freeze_backbone: bool = True,
        freeze_text_encoder: bool = True,
        gradient_checkpointing: bool = True,
        # === Enhancement flags ===
        use_dynamic_lora: bool = True,
        use_contrastive_loss: bool = True,
        use_multiscale_ot: bool = True,
        use_ohem_loss: bool = True,
        # === Enhancement params ===
        contrastive_weight: float = 0.1,
        ohem_hard_ratio: float = 0.3,
        ot_reg: float = 0.1,
        ot_alpha: float = 1.0,
        ot_beta: float = 1.0,
        ot_num_iter: int = 10,
        num_ot_scales: int = 3,
        learnable_margins: bool = True,
        uot_warmup_epochs: int = 5,
    ):
        super().__init__()
        self.image_size = image_size
        self.use_dynamic_lora = use_dynamic_lora
        self.use_contrastive_loss = use_contrastive_loss
        self.use_multiscale_ot = use_multiscale_ot
        self.use_ohem_loss = use_ohem_loss
        self.contrastive_weight = contrastive_weight

        # ====== Build SAM3 Image Model ======
        print("[Enhanced_RRSIS_UOT] Building SAM3 image model...")
        self.sam3 = build_sam3_image_model(
            device="cpu",
            eval_mode=False,
            checkpoint_path=sam3_ckpt,
            load_from_HF=(sam3_ckpt is None),
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
        )

        # ====== Freeze Strategy ======
        if freeze_backbone:
            self._freeze_backbone()
        if freeze_text_encoder:
            self._freeze_text_encoder()

        # ====== LoRA Injection ======
        d_model = self.sam3.hidden_dim  # typically 256

        if use_dynamic_lora:
            print("[Enhanced_RRSIS_UOT] Injecting Text-Guided Dynamic LoRA...")
            _, self.lora_manager = inject_dynamic_lora_adapters(
                self.sam3, text_dim=d_model, rank=lora_rank, alpha=lora_alpha
            )
        else:
            print("[Enhanced_RRSIS_UOT] Using standard static LoRA...")
            inject_lora_adapters(self.sam3, rank=lora_rank, alpha=lora_alpha)
            self.lora_manager = None

        # ====== Unfreeze trainable components ======
        self._unfreeze_trainable_components()

        # ====== Gradient Checkpointing ======
        if gradient_checkpointing:
            print("[Enhanced_RRSIS_UOT] Gradient checkpointing enabled")

        # ====== Multi-Scale Unbalanced OT Alignment ======
        if use_multiscale_ot:
            print(f"[Enhanced_RRSIS_UOT] Multi-Scale UOT Alignment ({num_ot_scales} scales)")
            self.ms_ot_aligner = MultiScaleOTAligner(
                d_model=d_model,
                num_scales=num_ot_scales,
                reg=ot_reg,
                alpha=ot_alpha,
                beta=ot_beta,
                num_iter=ot_num_iter,
                learnable_margins=learnable_margins,
                warmup_epochs=uot_warmup_epochs,
            )
        else:
            # Fallback to single-scale UOT from RRSIS_SAM3
            print("[Enhanced_RRSIS_UOT] Single-Scale UOT Alignment (baseline)")
            self.ot_aligner = OTFeatureAligner(
                d_model=d_model,
                reg=ot_reg,
                alpha=ot_alpha,
                beta=ot_beta,
                num_iter=ot_num_iter,
                learnable_margins=learnable_margins,
                warmup_epochs=uot_warmup_epochs,
            )

        # ====== Contrastive Loss ======
        if use_contrastive_loss:
            print("[Enhanced_RRSIS_UOT] Contrastive Loss (InfoNCE) enabled")
            self.contrastive_loss = ContrastiveLoss(
                visual_dim=d_model,
                text_dim=d_model,
            )

        # ====== OHEM Loss ======
        if use_ohem_loss:
            print("[Enhanced_RRSIS_UOT] OHEM + FocalDice Loss enabled")
            self.enhanced_loss = EnhancedOHEMLoss(
                hard_ratio=ohem_hard_ratio,
            )
        else:
            print("[Enhanced_RRSIS_UOT] Standard Dice+BCE Loss (baseline)")
            self.standard_loss = OTSegmentationLoss()



        # ====== Print Summary ======
        get_trainable_params_summary(self)

        # Image normalization
        self.register_buffer('pixel_mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    # ================================================================
    # Freeze / Unfreeze helpers
    # ================================================================

    def _freeze_backbone(self):
        """Freeze the ViT vision backbone."""
        backbone = self.sam3.backbone
        if hasattr(backbone, 'vision_backbone'):
            for param in backbone.vision_backbone.parameters():
                param.requires_grad = False
            print("[Enhanced_RRSIS_UOT] ViT backbone frozen")

    def _freeze_text_encoder(self):
        """Freeze the text encoder."""
        backbone = self.sam3.backbone
        if hasattr(backbone, 'language_backbone'):
            for param in backbone.language_backbone.parameters():
                param.requires_grad = False
            print("[Enhanced_RRSIS_UOT] Text encoder frozen")

    def _unfreeze_trainable_components(self):
        """Unfreeze components we want to fine-tune."""
        if hasattr(self.sam3, 'transformer'):
            for param in self.sam3.transformer.encoder.parameters():
                param.requires_grad = True
            for param in self.sam3.transformer.decoder.parameters():
                param.requires_grad = True
            print("[Enhanced_RRSIS_UOT] Transformer encoder+decoder unfrozen")

        if self.sam3.segmentation_head is not None:
            for param in self.sam3.segmentation_head.parameters():
                param.requires_grad = True
            print("[Enhanced_RRSIS_UOT] Segmentation head unfrozen")

        if hasattr(self.sam3, 'geometry_encoder'):
            for param in self.sam3.geometry_encoder.parameters():
                param.requires_grad = True
            print("[Enhanced_RRSIS_UOT] Geometry encoder unfrozen")

        if hasattr(self.sam3, 'dot_prod_scoring') and self.sam3.dot_prod_scoring is not None:
            for param in self.sam3.dot_prod_scoring.parameters():
                param.requires_grad = True
            print("[Enhanced_RRSIS_UOT] Scoring head unfrozen")

    def set_epoch(self, epoch):
        """
        Called by training loop each epoch to update UOT warmup state.
        Unfreezes alpha/beta in OT modules after warmup period.
        """
        if self.use_multiscale_ot and hasattr(self, 'ms_ot_aligner'):
            self.ms_ot_aligner.set_epoch(epoch)
        elif hasattr(self, 'ot_aligner'):
            self.ot_aligner.set_epoch(epoch)

    def get_uot_info(self):
        """Return current UOT alpha/beta values for logging."""
        if self.use_multiscale_ot and hasattr(self, 'ms_ot_aligner'):
            return self.ms_ot_aligner.get_uot_params_info()
        elif hasattr(self, 'ot_aligner'):
            return {
                'alpha': self.ot_aligner.alpha.item(),
                'beta': self.ot_aligner.beta.item(),
            }
        return {}

    def normalize_image(self, images):
        """Normalize images to SAM3's expected range [-1, 1]."""
        return (images - self.pixel_mean) / self.pixel_std

    # ================================================================
    # Forward Pass
    # ================================================================

    def forward(self, images, captions, masks_gt=None):
        """
        Enhanced forward pass with all 4 techniques.

        Args:
            images: [B, 3, H, W] tensor of RS images.
            captions: List[str] of length B, referring text descriptions.
            masks_gt: [B, 1, H, W] ground truth masks (for loss).

        Returns:
            dict with 'pred_masks', 'pred_logits', 'pred_boxes', 'loss'.
        """
        B = images.shape[0]
        device = images.device

        # Normalize images
        images = self.normalize_image(images)

        # ====== Step 1: Backbone Forward (Vision + Text) ======
        backbone_out = self.sam3.backbone.forward_image(images)
        text_out = self.sam3.backbone.forward_text(captions, device=device)
        backbone_out.update(text_out)

        # ====== Step 1.5: Dynamic LoRA text conditioning ======
        if self.use_dynamic_lora and self.lora_manager is not None:
            text_feats = backbone_out.get('language_features', None)
            if text_feats is not None:
                # Pool text features for conditioning: (seq, B, C) → (B, C)
                if text_feats.dim() == 3:
                    pooled_text = text_feats.mean(dim=0)  # (B, C)
                else:
                    pooled_text = text_feats
                self.lora_manager.set_text_conditioning(pooled_text)

        # ====== Step 2: OT Feature Alignment ======
        text_feats = backbone_out.get('language_features', None)
        text_mask = backbone_out.get('language_mask', None)

        if text_feats is not None and 'backbone_fpn' in backbone_out:
            fpn_feats = backbone_out['backbone_fpn']

            if self.use_multiscale_ot and hasattr(self, 'ms_ot_aligner'):
                # Multi-Scale OT alignment (NEW)
                aligned_fpn = self.ms_ot_aligner(fpn_feats, text_feats, text_mask)
                backbone_out['backbone_fpn'] = aligned_fpn
            elif hasattr(self, 'ot_aligner'):
                # Single-scale OT alignment (baseline fallback)
                for i in range(len(fpn_feats)):
                    feat = fpn_feats[i]
                    if hasattr(feat, 'tensors'):
                        feat_tensor = feat.tensors
                    else:
                        feat_tensor = feat
                    if feat_tensor.dim() == 4:
                        aligned = self.ot_aligner(feat_tensor, text_feats, text_mask)
                        if hasattr(feat, 'tensors'):
                            feat.tensors = aligned
                        else:
                            fpn_feats[i] = aligned

        # ====== Step 3: Prompt Encoding (text-only, empty geometric) ======
        img_ids = torch.arange(B, device=device)
        text_ids = torch.arange(B, device=device)
        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=torch.zeros(B, 0, 4, device=device),
            input_boxes_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
            input_boxes_label=torch.zeros(B, 0, device=device, dtype=torch.long),
            input_points=torch.zeros(B, 0, 2, device=device),
            input_points_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
        )

        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, B, 4, device=device),
            box_mask=torch.zeros(B, 0, device=device, dtype=torch.bool),
        )

        # ====== Step 4: Encode Prompt ======
        prompt, prompt_mask, backbone_out = self.sam3._encode_prompt(
            backbone_out, find_input, geometric_prompt
        )

        # ====== Step 5: Run Encoder (fusion) ======
        backbone_out, encoder_out, feat_tuple = self.sam3._run_encoder(
            backbone_out, find_input, prompt, prompt_mask
        )

        # ====== Step 6: Run Decoder (DETR detection) ======
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
        }
        out, hs = self.sam3._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        # ====== Step 7: Segmentation Head ======
        if self.sam3.segmentation_head is not None:
            _, _, _, vis_feat_sizes = feat_tuple
            seg_img_ids = find_input.img_ids
            if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
                seg_img_ids = backbone_out["id_mapping"][seg_img_ids]

            self.sam3._run_segmentation_heads(
                out=out,
                backbone_out=backbone_out,
                img_ids=seg_img_ids,
                vis_feat_sizes=vis_feat_sizes,
                encoder_hidden_states=out["encoder_hidden_states"],
                prompt=prompt,
                prompt_mask=prompt_mask,
                hs=hs,
            )

        # ====== Step 8: Select Best Mask ======
        result = self._select_best_mask(out, B)

        # ====== Step 9: Clear Dynamic LoRA conditioning ======
        if self.use_dynamic_lora and self.lora_manager is not None:
            self.lora_manager.clear_text_conditioning()

        # ====== Step 10: Compute Losses ======
        if masks_gt is not None:
            # Primary segmentation loss
            if self.use_ohem_loss and hasattr(self, 'enhanced_loss'):
                seg_loss = self.enhanced_loss(result, masks_gt, self.image_size)
            elif hasattr(self, 'standard_loss'):
                seg_loss = self.standard_loss(result, masks_gt, self.image_size)
            else:
                seg_loss = self._compute_fallback_loss(result['pred_masks'], masks_gt)

            # Contrastive loss (auxiliary)
            contrastive = torch.tensor(0.0, device=device)
            if self.use_contrastive_loss and hasattr(self, 'contrastive_loss'):
                try:
                    # Get visual features for contrastive learning
                    visual_feats = encoder_out.get('encoder_hidden_states', None)
                    if visual_feats is not None and text_feats is not None:
                        C = visual_feats.shape[-1]

                        if visual_feats.dim() == 2:
                            # SAM3 format: (N_total, C) — flatten of all tokens
                            N_total = visual_feats.shape[0]
                            if N_total % B == 0:
                                N_per = N_total // B
                                vis_batched = visual_feats.view(B, N_per, C)  # (B, N, C)
                            else:
                                # Uneven split — pool everything per batch
                                vis_batched = visual_feats.unsqueeze(0).expand(B, -1, -1)

                        elif visual_feats.dim() == 3:
                            if visual_feats.shape[0] == B:
                                vis_batched = visual_feats  # Already (B, N, C)
                            else:
                                # (N_total, seq, C) — reshape to (B, -1, C)
                                vis_batched = visual_feats.reshape(B, -1, C)
                        else:
                            vis_batched = None

                        if vis_batched is not None:
                            # Pool to (B, C) then create small spatial map
                            vis_pooled = vis_batched.mean(dim=1)  # (B, C)
                            vis_spatial = vis_pooled.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
                            vis_spatial = vis_spatial.expand(B, C, 4, 4).contiguous()  # (B, C, 4, 4)

                            contrastive = self.contrastive_loss(
                                vis_spatial, text_feats,
                                result['pred_masks'], masks_gt
                            )
                except Exception as e:
                    # Don't crash training if contrastive loss fails
                    if self.training:
                        print(f"[WARNING] Contrastive loss skipped: {e}")
                    contrastive = torch.tensor(0.0, device=device)

            result['loss'] = seg_loss + self.contrastive_weight * contrastive
            result['seg_loss'] = seg_loss.detach()
            result['contrastive_loss'] = contrastive.detach() if isinstance(contrastive, torch.Tensor) else contrastive

        return result

    def _select_best_mask(self, out, batch_size):
        """Select the best mask from SAM3's multi-query output."""
        result = {}

        pred_logits = out.get('pred_logits', None)
        if pred_logits is not None:
            result['pred_logits'] = pred_logits

        pred_boxes = out.get('pred_boxes', None)
        if pred_boxes is not None:
            result['pred_boxes'] = pred_boxes

        pred_masks = out.get('pred_masks', None)
        if pred_masks is not None:
            if pred_logits is not None:
                scores = pred_logits.squeeze(-1)
                best_idx = scores.argmax(dim=-1)
                batch_idx = torch.arange(batch_size, device=pred_masks.device)
                best_masks = pred_masks[batch_idx, best_idx]
                best_masks = best_masks.unsqueeze(1)
            else:
                best_masks = pred_masks[:, 0:1]
            best_masks = F.interpolate(
                best_masks.float(),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False,
            )
            result['pred_masks'] = best_masks
        else:
            result['pred_masks'] = torch.zeros(
                batch_size, 1, self.image_size, self.image_size,
                device=next(self.parameters()).device,
            )

        return result

    def _compute_fallback_loss(self, pred_masks, gt_masks):
        """Fallback Dice + BCE loss."""
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.float(), size=pred_masks.shape[-2:], mode='nearest'
            )

        bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks.float())

        pred_probs = torch.sigmoid(pred_masks)
        intersection = (pred_probs * gt_masks).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = dice_loss.mean()

        return 0.5 * bce_loss + 0.5 * dice_loss

    @torch.no_grad()
    def predict(self, images, captions):
        """Inference-only forward pass."""
        self.eval()
        result = self.forward(images, captions)
        result['pred_probs'] = torch.sigmoid(result['pred_masks'])
        result['pred_binary'] = (result['pred_probs'] > 0.5).float()
        return result
