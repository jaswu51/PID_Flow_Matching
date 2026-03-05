"""
Base Class for Generative Methods

This module defines the abstract interface that generative methods
(e.g., DDPM) must implement. This ensures consistency across different
implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn


class BaseMethod(nn.Module, ABC):
    """
    Abstract base class for generative modeling methods.

    Methods (e.g., DDPM) should inherit from this class
    and implement the required methods.

    Attributes:
        model: The neural network (typically a U-Net) used for prediction
        device: Device to run computations on
    """

    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the method.

        Args:
            model: Neural network for prediction (e.g., UNet)
            device: Device to use for computations
        """
        super().__init__()
        self.model = model
        self.device = device
    
    @abstractmethod
    def compute_loss(
        self, 
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the training loss for a batch of data.
        
        Args:
            x: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """
        pass
    
    @abstractmethod
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        pass
    
    def train_mode(self):
        """Set the model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set the model to evaluation mode."""
        self.model.eval()
    
    def to(self, device: torch.device) -> 'BaseMethod':
        """
        Move the method to a device.
        
        Args:
            device: Target device
        
        Returns:
            self for chaining
        """
        super().to(device)
        self.device = device
        return self
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self.model.parameters()
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Get the state dict for checkpointing.
        
        Returns:
            Dictionary containing method state
        """
        return {
            'model': self.model.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load a state dict from a checkpoint.
        
        Args:
            state_dict: State dict to load
        """
        self.model.load_state_dict(state_dict['model'])
