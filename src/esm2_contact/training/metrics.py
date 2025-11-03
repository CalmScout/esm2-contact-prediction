"""
Evaluation Metrics for Binary Contact Prediction

This module provides comprehensive evaluation metrics for binary protein contact
prediction, including standard classification metrics and contact-specific metrics
like Precision@L.

Key Features:
- AUC (Area Under ROC Curve) calculation
- Precision@L and Precision@L5 for contact prediction
- Standard classification metrics (precision, recall, F1)
- Safe calculations with error handling for edge cases
- Support for variable-length proteins with masking

Metrics Included:
- AUC: Area under ROC curve
- Precision@L: Precision for top-L predictions
- Precision@L5: Precision for top-5L predictions
- Precision, Recall, F1-score
- Matthews Correlation Coefficient (MCC)

Usage:
    metrics = ContactMetrics()
    results = metrics.calculate_all_metrics(predictions, targets, lengths)
    print(f"AUC: {results['auc']:.4f}, Prec@L: {results['precision_at_l']:.4f}")
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (auc, average_precision_score, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)


class ContactMetrics:
    """
    Comprehensive metrics for binary contact prediction evaluation.

    This class provides methods to calculate various metrics for evaluating
    binary contact maps, with special attention to contact-specific metrics
    like Precision@L which are standard in protein structure prediction.
    """

    def __init__(self, binary_threshold: float = 0.5):
        """
        Initialize ContactMetrics.

        Args:
            binary_threshold (float): Threshold for converting probabilities to binary (default: 0.5)
        """
        self.binary_threshold = binary_threshold
        self.reset_incremental_metrics()

    def reset_incremental_metrics(self):
        """Reset incremental metrics for new evaluation."""
        self.batch_metrics = []
        self.total_samples = 0

    def update_batch(self, predictions: torch.Tensor, targets: torch.Tensor,
                    lengths: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Update incremental metrics with a new batch.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits (batch_size, H, W)
            targets (torch.Tensor): Target binary labels (batch_size, H, W)
            lengths (torch.Tensor): Original sequence lengths (batch_size,)
            mask (Optional[torch.Tensor]): Mask for valid regions (batch_size, H, W)
        """
        # Calculate metrics for this batch immediately
        batch_metrics = self.calculate_all_metrics(predictions, targets, lengths, mask)
        self.batch_metrics.append(batch_metrics)
        self.total_samples += predictions.shape[0]

    def calculate_incremental_metrics(self) -> Dict[str, float]:
        """
        Calculate final metrics from all batch metrics.

        Returns:
            Dict[str, float]: Dictionary containing all calculated metrics
        """
        if self.total_samples == 0 or not self.batch_metrics:
            return {
                'auc': 0.5, 'precision_at_l': 0.0, 'precision_at_l5': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mcc': 0.0, 'accuracy': 0.0
            }

        try:
            # Aggregate metrics across batches (simple average)
            final_metrics = {}
            for key in self.batch_metrics[0].keys():
                final_metrics[key] = sum(m[key] for m in self.batch_metrics) / len(self.batch_metrics)

            return final_metrics

        except Exception as e:
            warnings.warn(f"Incremental metrics calculation error: {e}")
            return {
                'auc': 0.5, 'precision_at_l': 0.0, 'precision_at_l5': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'mcc': 0.0, 'accuracy': 0.0
            }

    def calculate_auc_safe(self, predictions: torch.Tensor, targets: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> float:
        """
        Calculate AUC with safe error handling.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits
            targets (torch.Tensor): Target binary labels
            mask (Optional[torch.Tensor]): Mask for valid regions

        Returns:
            float: AUC score (0.5 for random, 1.0 for perfect)
        """
        try:
            # Convert logits to probabilities if needed
            if (predictions < 0).any() or (predictions > 1).any():
                predictions = torch.sigmoid(predictions)

            # Flatten arrays
            pred_flat = predictions.flatten().cpu().numpy()
            target_flat = targets.flatten().cpu().numpy()

            # Apply mask if provided
            if mask is not None:
                mask_flat = mask.flatten().cpu().numpy()
                pred_flat = pred_flat[mask_flat > 0]
                target_flat = target_flat[mask_flat > 0]

            # Remove any invalid targets
            valid_mask = (target_flat >= 0) & (target_flat <= 1)
            pred_valid = pred_flat[valid_mask]
            target_valid = target_flat[valid_mask]

            if len(pred_valid) == 0:
                return 0.5

            # Check if we have both classes
            unique_targets = np.unique(target_valid)
            if len(unique_targets) < 2:
                # If only one class, return baseline
                if unique_targets[0] == 0:
                    return 0.5  # All negative
                else:
                    return 1.0  # All positive

            # Calculate AUC
            auc_score = roc_auc_score(target_valid, pred_valid)
            return max(0.0, min(1.0, auc_score))

        except Exception as e:
            warnings.warn(f"AUC calculation error: {e}")
            return 0.5

    def calculate_precision_safe(self, predictions: torch.Tensor, targets: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> float:
        """
        Calculate precision with safe error handling.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits
            targets (torch.Tensor): Target binary labels
            mask (Optional[torch.Tensor]): Mask for valid regions

        Returns:
            float: Precision score (0.0 to 1.0)
        """
        try:
            # Convert to binary predictions
            if (predictions < 0).any() or (predictions > 1).any():
                predictions = torch.sigmoid(predictions)

            pred_binary = (predictions > self.binary_threshold).float()

            # Flatten arrays
            pred_flat = pred_binary.flatten().cpu().numpy()
            target_flat = targets.flatten().cpu().numpy()

            # Apply mask if provided
            if mask is not None:
                mask_flat = mask.flatten().cpu().numpy()
                pred_flat = pred_flat[mask_flat > 0]
                target_flat = target_flat[mask_flat > 0]

            # Calculate precision with zero division handling
            precision = precision_score(target_flat, pred_flat, zero_division=0)
            return max(0.0, min(1.0, precision))

        except Exception as e:
            warnings.warn(f"Precision calculation error: {e}")
            return 0.0

    def calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> float:
        """
        Calculate accuracy with safe error handling.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits
            targets (torch.Tensor): Target binary labels
            mask (Optional[torch.Tensor]): Mask for valid regions

        Returns:
            float: Accuracy score (0.0 to 1.0)
        """
        try:
            # Convert logits to probabilities if needed
            if (predictions < 0).any() or (predictions > 1).any():
                predictions = torch.sigmoid(predictions)

            # Convert to binary predictions
            pred_binary = (predictions > self.binary_threshold).float()

            # Flatten arrays
            pred_flat = pred_binary.flatten().cpu().numpy()
            target_flat = targets.flatten().cpu().numpy()

            # Apply mask if provided
            if mask is not None:
                mask_flat = mask.flatten().cpu().numpy()
                pred_flat = pred_flat[mask_flat > 0]
                target_flat = target_flat[mask_flat > 0]

            # Remove any invalid targets
            valid_mask = (target_flat >= 0) & (target_flat <= 1)
            pred_valid = pred_flat[valid_mask]
            target_valid = target_flat[valid_mask]

            if len(pred_valid) == 0:
                return 0.0

            # Calculate accuracy using sklearn
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(target_valid, pred_valid)

            return float(accuracy)

        except Exception as e:
            warnings.warn(f"Accuracy calculation error: {e}")
            return 0.0

    def calculate_precision_at_l(self, predictions: torch.Tensor, targets: torch.Tensor,
                                 lengths: torch.Tensor, k: int = 1) -> float:
        """
        Calculate Precision@L metric for contact prediction.

        Precision@L measures the precision of the top-k*L predictions, where L is
        the sequence length. This is a standard metric in protein contact prediction.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits (batch_size, H, W)
            targets (torch.Tensor): Target binary labels (batch_size, H, W)
            lengths (torch.Tensor): Original sequence lengths (batch_size,)
            k (int): Multiplier for L (default: 1 for Precision@L)

        Returns:
            float: Precision@L score (0.0 to 1.0)
        """
        try:
            total_correct = 0
            total_predicted = 0

            # Convert logits to probabilities if needed
            if (predictions < 0).any() or (predictions > 1).any():
                predictions = torch.sigmoid(predictions)

            for i, L in enumerate(lengths):
                if L < 2:
                    continue

                # Get upper triangle (excluding diagonal)
                pred_matrix = predictions[i, :L, :L]
                target_matrix = targets[i, :L, :L]

                rows, cols = torch.triu_indices(L, L, offset=1)
                pred_values = pred_matrix[rows, cols]
                target_values = target_matrix[rows, cols]

                # Skip if no positive targets
                if torch.sum(target_values) == 0:
                    continue

                # Top-k*L predictions
                num_predictions = min(k * L, len(target_values))
                if num_predictions > 0:
                    _, top_indices = torch.topk(pred_values, num_predictions)
                    correct = target_values[top_indices].sum().item()
                    total_correct += correct
                    total_predicted += num_predictions

            return total_correct / max(total_predicted, 1)

        except Exception as e:
            warnings.warn(f"Precision@L calculation error: {e}")
            return 0.0

    def calculate_standard_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                                  mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate standard classification metrics.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits
            targets (torch.Tensor): Target binary labels
            mask (Optional[torch.Tensor]): Mask for valid regions

        Returns:
            Dict[str, float]: Dictionary containing precision, recall, f1, mcc
        """
        try:
            # Convert to binary predictions
            if (predictions < 0).any() or (predictions > 1).any():
                predictions = torch.sigmoid(predictions)

            pred_binary = (predictions > self.binary_threshold).float()

            # Flatten arrays
            pred_flat = pred_binary.flatten().cpu().numpy()
            target_flat = targets.flatten().cpu().numpy()

            # Apply mask if provided
            if mask is not None:
                mask_flat = mask.flatten().cpu().numpy()
                pred_flat = pred_flat[mask_flat > 0]
                target_flat = target_flat[mask_flat > 0]

            # Calculate metrics
            precision = precision_score(target_flat, pred_flat, zero_division=0)
            recall = recall_score(target_flat, pred_flat, zero_division=0)
            f1 = f1_score(target_flat, pred_flat, zero_division=0)

            # Matthews correlation coefficient
            try:
                mcc = matthews_corrcoef(target_flat, pred_flat)
            except:
                mcc = 0.0

            # Average precision score
            try:
                ap = average_precision_score(target_flat, pred_flat)
            except:
                ap = 0.0

            # Accuracy
            try:
                accuracy = accuracy_score(target_flat, pred_flat)
            except:
                accuracy = 0.0

            return {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'mcc': float(mcc),
                'average_precision': float(ap),
                'accuracy': float(accuracy)
            }

        except Exception as e:
            warnings.warn(f"Standard metrics calculation error: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mcc': 0.0,
                'average_precision': 0.0,
                'accuracy': 0.0
            }

    def calculate_all_metrics(self, predictions: torch.Tensor, targets: torch.Tensor,
                             lengths: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Calculate all available metrics.

        Args:
            predictions (torch.Tensor): Predicted probabilities or logits (batch_size, H, W)
            targets (torch.Tensor): Target binary labels (batch_size, H, W)
            lengths (torch.Tensor): Original sequence lengths (batch_size,)
            mask (Optional[torch.Tensor]): Mask for valid regions (batch_size, H, W)

        Returns:
            Dict[str, float]: Dictionary containing all calculated metrics
        """
        metrics = {}

        # Contact-specific metrics
        metrics['auc'] = self.calculate_auc_safe(predictions, targets, mask)
        metrics['precision_at_l'] = self.calculate_precision_at_l(predictions, targets, lengths, k=1)
        metrics['precision_at_l5'] = self.calculate_precision_at_l(predictions, targets, lengths, k=5)
        metrics['precision'] = self.calculate_precision_safe(predictions, targets, mask)

        # Standard classification metrics
        standard_metrics = self.calculate_standard_metrics(predictions, targets, mask)
        metrics.update(standard_metrics)

        return metrics

    def print_metrics(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """
        Print metrics in a formatted way.

        Args:
            metrics (Dict[str, float]): Metrics dictionary
            prefix (str): Prefix for printed lines (default: "")
        """
        print(f"{prefix}📊 Contact Prediction Metrics:")
        print(f"{prefix}   AUC:               {metrics.get('auc', 0.0):.4f}")
        print(f"{prefix}   Precision@L:       {metrics.get('precision_at_l', 0.0):.4f}")
        print(f"{prefix}   Precision@L5:      {metrics.get('precision_at_l5', 0.0):.4f}")
        print(f"{prefix}   Precision:         {metrics.get('precision', 0.0):.4f}")
        print(f"{prefix}   Recall:            {metrics.get('recall', 0.0):.4f}")
        print(f"{prefix}   F1-score:          {metrics.get('f1', 0.0):.4f}")
        print(f"{prefix}   MCC:               {metrics.get('mcc', 0.0):.4f}")
        print(f"{prefix}   Accuracy:          {metrics.get('accuracy', 0.0):.4f}")
        print(f"{prefix}   Average Precision: {metrics.get('average_precision', 0.0):.4f}")


def calculate_batch_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                           lengths: torch.Tensor, mask: Optional[torch.Tensor] = None,
                           binary_threshold: float = 0.5) -> Dict[str, float]:
    """
    Convenience function to calculate metrics for a batch of predictions.

    Args:
        predictions (torch.Tensor): Predicted probabilities or logits (batch_size, H, W)
        targets (torch.Tensor): Target binary labels (batch_size, H, W)
        lengths (torch.Tensor): Original sequence lengths (batch_size,)
        mask (Optional[torch.Tensor]): Mask for valid regions (batch_size, H, W)
        binary_threshold (float): Threshold for binary predictions (default: 0.5)

    Returns:
        Dict[str, float]: Dictionary containing all calculated metrics
    """
    calculator = ContactMetrics(binary_threshold=binary_threshold)
    return calculator.calculate_all_metrics(predictions, targets, lengths, mask)


# Test function for metrics
def test_metrics():
    """Test metrics calculation with synthetic data."""
    print("🧪 Testing Contact Metrics...")

    # Create synthetic data
    batch_size, height, width = 2, 64, 64
    logits = torch.randn(batch_size, height, width)
    targets = (torch.rand(batch_size, height, width) > 0.8).float()  # Very sparse contacts
    lengths = torch.tensor([50, 60])
    mask = torch.ones(batch_size, height, width).bool()

    print(f"   Synthetic data: logits {logits.shape}, targets {targets.shape}")
    print(f"   Target density: {targets.mean():.4f}")
    print(f"   Contact counts: {[torch.sum(targets[i, :l, :l]).item() for i, l in enumerate(lengths)]}")

    # Calculate metrics
    calculator = ContactMetrics(binary_threshold=0.5)
    metrics = calculator.calculate_all_metrics(logits, targets, lengths, mask)

    # Print results
    calculator.print_metrics(metrics, prefix="   ")

    # Test individual metric calculations
    auc = calculator.calculate_auc_safe(logits, targets, mask)
    prec_l = calculator.calculate_precision_at_l(logits, targets, lengths, k=1)
    precision = calculator.calculate_precision_safe(logits, targets, mask)

    print(f"✅ Individual metrics:")
    print(f"   AUC: {auc:.4f}")
    print(f"   Precision@L: {prec_l:.4f}")
    print(f"   Precision: {precision:.4f}")

    # Test edge cases
    print("   Testing edge cases...")

    # All zeros
    all_zeros = torch.zeros_like(targets)
    auc_zero = calculator.calculate_auc_safe(torch.randn_like(logits), all_zeros, mask)
    print(f"   All zeros AUC: {auc_zero:.4f}")

    # All ones
    all_ones = torch.ones_like(targets)
    auc_one = calculator.calculate_auc_safe(torch.randn_like(logits), all_ones, mask)
    print(f"   All ones AUC: {auc_one:.4f}")

    print("🎉 Metrics tests completed!")


if __name__ == "__main__":
    # Run metrics tests
    test_metrics()