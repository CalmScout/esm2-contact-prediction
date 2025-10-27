"""
Loss Functions for Binary Contact Prediction

This module provides loss functions specifically designed for binary protein contact
prediction with the CNN architecture. The losses handle class imbalance and provide
numerical stability for training.

Key Features:
- BCEWithLogitsLoss for numerical stability with raw logits
- Focal Loss for handling class imbalance
- Configurable positive class weighting
- Support for masked training (ignore padding regions)

Loss Functions:
- BCEWithLogitsLoss: Standard binary cross entropy with logits
- FocalLoss: Focal loss for hard example mining
- Weighted BCE: Class-weighted binary cross entropy

Usage:
    criterion = get_loss_function('bce', pos_weight=5.0)
    loss = criterion(logits, targets)  # logits are raw outputs (no sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in contact prediction.

    Focal loss adds a factor (1 - pt)^gamma to the standard cross entropy loss,
    which down-weights the loss assigned to well-classified examples and focuses
    training on hard examples.

    Args:
        alpha (float): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter (default: 2.0)
        pos_weight (Optional[float]): Positive class weight for BCE (default: None)
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0,
                 pos_weight: Optional[float] = None, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate focal loss.

        Args:
            inputs (torch.Tensor): Predicted logits (batch_size, H, W)
            targets (torch.Tensor): Target binary labels (batch_size, H, W)
            mask (Optional[torch.Tensor]): Mask for valid regions (batch_size, H, W)

        Returns:
            torch.Tensor: Calculated focal loss
        """
        # Calculate binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )

        # Apply mask if provided
        if mask is not None:
            bce_loss = bce_loss * mask

        # Calculate probability (pt)
        pt = torch.exp(-bce_loss)

        # Calculate focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            if mask is not None:
                return focal_loss.sum() / mask.sum()
            else:
                return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss with class balancing.

    This loss function allows for different weighting of positive and negative
    classes to address the inherent class imbalance in contact prediction
    (typically many more non-contacts than contacts).

    Args:
        pos_weight (float): Weight for positive class (default: 5.0)
        neg_weight (float): Weight for negative class (default: 1.0)
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """

    def __init__(self, pos_weight: float = 5.0, neg_weight: float = 1.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate weighted binary cross entropy loss.

        Args:
            inputs (torch.Tensor): Predicted logits (batch_size, H, W)
            targets (torch.Tensor): Target binary labels (batch_size, H, W)
            mask (Optional[torch.Tensor]): Mask for valid regions (batch_size, H, W)

        Returns:
            torch.Tensor: Calculated weighted BCE loss
        """
        # Calculate binary cross entropy with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Create weight tensor
        weights = torch.where(targets == 1, self.pos_weight, self.neg_weight)
        weighted_loss = bce_loss * weights

        # Apply mask if provided
        if mask is not None:
            weighted_loss = weighted_loss * mask

        # Apply reduction
        if self.reduction == 'mean':
            if mask is not None:
                return weighted_loss.sum() / mask.sum()
            else:
                return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:  # 'none'
            return weighted_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.

    Dice loss is based on the Dice coefficient, which is commonly used for
    evaluating binary segmentation. It's particularly useful when dealing
    with class imbalance.

    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
        reduction (str): Reduction method ('none', 'mean', 'sum') (default: 'mean')
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate Dice loss.

        Args:
            inputs (torch.Tensor): Predicted logits (batch_size, H, W)
            targets (torch.Tensor): Target binary labels (batch_size, H, W)
            mask (Optional[torch.Tensor]): Mask for valid regions (batch_size, H, W)

        Returns:
            torch.Tensor: Calculated Dice loss
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            probs_flat = probs_flat * mask_flat
            targets_flat = targets_flat * mask_flat

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Convert to loss
        dice_loss = 1.0 - dice

        return dice_loss


def get_loss_function(loss_type: str = 'bce', pos_weight: float = 5.0,
                      **kwargs) -> nn.Module:
    """
    Factory function to get a loss function for binary contact prediction.

    Args:
        loss_type (str): Type of loss function ('bce', 'focal', 'weighted_bce', 'dice')
        pos_weight (float): Weight for positive class (default: 5.0)
        **kwargs: Additional arguments for specific loss functions

    Returns:
        nn.Module: Configured loss function

    Raises:
        ValueError: If loss_type is not recognized
    """
    loss_type = loss_type.lower()

    if loss_type == 'bce':
        # Standard BCE with logits
        weight = torch.tensor(pos_weight)
        return nn.BCEWithLogitsLoss(pos_weight=weight, **kwargs)

    elif loss_type == 'focal':
        # Focal loss for class imbalance
        return FocalLoss(alpha=kwargs.get('alpha', 1.0),
                        gamma=kwargs.get('gamma', 2.0),
                        pos_weight=pos_weight,
                        reduction=kwargs.get('reduction', 'mean'))

    elif loss_type == 'weighted_bce':
        # Weighted BCE with separate positive/negative weights
        return WeightedBCELoss(pos_weight=pos_weight,
                              neg_weight=kwargs.get('neg_weight', 1.0),
                              reduction=kwargs.get('reduction', 'mean'))

    elif loss_type == 'dice':
        # Dice loss
        return DiceLoss(smooth=kwargs.get('smooth', 1e-6),
                       reduction=kwargs.get('reduction', 'mean'))

    elif loss_type == 'combined':
        # Combined BCE + Dice loss
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        dice_loss = DiceLoss(smooth=kwargs.get('smooth', 1e-6))

        class CombinedLoss(nn.Module):
            def __init__(self, bce_weight=0.7, dice_weight=0.3):
                super().__init__()
                self.bce_loss = bce_loss
                self.dice_loss = dice_loss
                self.bce_weight = bce_weight
                self.dice_weight = dice_weight

            def forward(self, inputs, targets, mask=None):
                bce = self.bce_loss(inputs, targets)
                dice = self.dice_loss(inputs, targets, mask)
                return self.bce_weight * bce + self.dice_weight * dice

        return CombinedLoss(
            bce_weight=kwargs.get('bce_weight', 0.7),
            dice_weight=kwargs.get('dice_weight', 0.3)
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Supported types: 'bce', 'focal', 'weighted_bce', 'dice', 'combined'")


def calculate_loss_weights(targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    """
    Calculate optimal positive class weight based on class distribution.

    Args:
        targets (torch.Tensor): Target binary labels
        mask (Optional[torch.Tensor]): Mask for valid regions

    Returns:
        float: Calculated positive class weight
    """
    if mask is not None:
        valid_targets = targets[mask]
    else:
        valid_targets = targets

    if len(valid_targets) == 0:
        return 1.0

    pos_count = (valid_targets == 1).sum().float()
    neg_count = (valid_targets == 0).sum().float()

    if pos_count == 0:
        return 1.0

    # Weight = negative_count / positive_count
    weight = neg_count / pos_count
    return min(weight.item(), 10.0)  # Cap at 10 to avoid extreme weights


# Test function for loss functions
def test_loss_functions():
    """Test various loss functions with synthetic data."""
    print("🧪 Testing Loss Functions...")

    # Create synthetic data
    batch_size, height, width = 2, 64, 64
    logits = torch.randn(batch_size, height, width)
    targets = (torch.rand(batch_size, height, width) > 0.7).float()  # Sparse contacts
    mask = torch.ones(batch_size, height, width).bool()

    print(f"   Synthetic data: logits {logits.shape}, targets {targets.shape}")
    print(f"   Target density: {targets.mean():.4f}")

    # Test different loss functions
    loss_types = ['bce', 'focal', 'weighted_bce', 'dice', 'combined']

    for loss_type in loss_types:
        try:
            criterion = get_loss_function(loss_type, pos_weight=5.0)
            loss = criterion(logits, targets, mask)
            print(f"✅ {loss_type:12s}: loss = {loss.item():.4f}")
        except Exception as e:
            print(f"❌ {loss_type:12s}: error - {e}")

    # Test automatic weight calculation
    auto_weight = calculate_loss_weights(targets, mask)
    print(f"✅ Auto weight: {auto_weight:.2f}")

    print("🎉 Loss function tests completed!")


if __name__ == "__main__":
    # Run loss function tests
    test_loss_functions()