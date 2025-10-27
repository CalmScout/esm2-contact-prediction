"""
Binary CNN Model for Protein Contact Prediction

This module implements a memory-efficient CNN architecture for protein contact prediction
with binary output format (0/1 contact maps) as required by the problem definition.

Key Differences from Probability-Based Models:
- No sigmoid activation in final layer
- Outputs raw logits for BCEWithLogitsLoss (numerical stability)
- Binary prediction via thresholding during inference
- Optimized for small dataset training

Architecture:
- Input: 68-channel tensors (4 template + 64 ESM2 channels)
- 3 convolutional blocks with batch normalization and ReLU
- 1x1 convolution for final prediction (no activation)
- Output: Raw logits converted to binary via threshold

Usage:
    model = BinaryContactCNN(in_channels=68, base_channels=32)

    # Training (raw logits)
    logits = model(features)
    loss = criterion(logits, targets)  # BCEWithLogitsLoss

    # Inference (binary)
    binary_contacts = model.predict_binary(features, threshold=0.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryContactCNN(nn.Module):
    """
    Memory-efficient 3-layer CNN for binary protein contact prediction.

    This model outputs raw logits (not probabilities) for numerical stability
    when used with BCEWithLogitsLoss. Binary predictions are obtained via
    thresholding during inference.

    Args:
        in_channels (int): Number of input channels (default: 68)
        base_channels (int): Base number of channels (default: 32)
        dropout_rate (float): Dropout rate (default: 0.1)
    """

    def __init__(self, in_channels=68, base_channels=32, dropout_rate=0.1):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.dropout_rate = dropout_rate

        # 3 convolutional blocks with increasing channel counts
        self.conv1 = self._make_block(in_channels, base_channels)
        self.conv2 = self._make_block(base_channels, base_channels * 2)
        self.conv3 = self._make_block(base_channels * 2, base_channels * 4)

        # Final prediction layer - NO sigmoid activation
        self.predictor = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(base_channels * 2, 1, 1)  # 1x1 convolution, NO activation
        )

        # Initialize weights
        self._initialize_weights()

    def _make_block(self, in_ch, out_ch):
        """Create a convolutional block with batch normalization."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass - outputs raw logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Raw logits of shape (batch_size, H, W)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        logits = self.predictor(x).squeeze(1)  # Remove channel dimension
        return logits

    def predict_binary(self, x, threshold=0.5):
        """
        Binary prediction for inference.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)
            threshold (float): Threshold for binary decision (default: 0.5)

        Returns:
            torch.Tensor: Binary contact map of shape (batch_size, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
            binary_contacts = (probabilities > threshold).float()
        return binary_contacts

    def predict_probabilities(self, x):
        """
        Probability prediction for evaluation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Contact probabilities of shape (batch_size, H, W)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities

    def _initialize_weights(self):
        """Initialize model weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_model_info(self):
        """Get model information dictionary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'architecture': 'BinaryContactCNN',
            'input_channels': self.in_channels,
            'base_channels': self.base_channels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'dropout_rate': self.dropout_rate,
            'memory_footprint_mb': total_params * 4 / 1024**2  # Assuming float32
        }

    def save_model(self, filepath):
        """Save model state dict and configuration."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'in_channels': self.in_channels,
                'base_channels': self.base_channels,
                'dropout_rate': self.dropout_rate
            },
            'model_info': self.get_model_info()
        }, filepath)

    @classmethod
    def load_model(cls, filepath, device='cpu'):
        """Load model from saved checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


def create_model(config=None):
    """
    Factory function to create a BinaryContactCNN model with configuration.

    Args:
        config (dict): Configuration dictionary with model parameters

    Returns:
        BinaryContactCNN: Configured model instance
    """
    if config is None:
        config = {
            'in_channels': 68,
            'base_channels': 32,
            'dropout_rate': 0.1
        }

    return BinaryContactCNN(**config)


# Test function for model validation
def test_model():
    """Test model functionality and architecture."""
    print("🧪 Testing BinaryContactCNN Model...")

    # Create model
    model = BinaryContactCNN(in_channels=68, base_channels=32)

    # Test forward pass
    batch_size, channels, height, width = 2, 68, 128, 128
    dummy_input = torch.randn(batch_size, channels, height, width)

    model.eval()
    with torch.no_grad():
        # Test raw logits output
        logits = model(dummy_input)
        print(f"✅ Logits output shape: {logits.shape}")
        print(f"   Expected: ({batch_size}, {height}, {width})")

        # Test binary prediction
        binary_output = model.predict_binary(dummy_input, threshold=0.5)
        print(f"✅ Binary output shape: {binary_output.shape}")
        print(f"   Unique values: {torch.unique(binary_output).tolist()}")

        # Test probability prediction
        prob_output = model.predict_probabilities(dummy_input)
        print(f"✅ Probability output shape: {prob_output.shape}")
        print(f"   Value range: [{prob_output.min():.4f}, {prob_output.max():.4f}]")

    # Model info
    model_info = model.get_model_info()
    print(f"✅ Model information:")
    for key, value in model_info.items():
        print(f"   {key}: {value}")

    print("🎉 Model test completed successfully!")
    return model


if __name__ == "__main__":
    # Run model test
    test_model()