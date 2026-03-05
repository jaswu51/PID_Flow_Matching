"""
Exponential Moving Average (EMA) for Model Parameters

EMA maintains a shadow copy of model parameters that are updated as an
exponential moving average of the training parameters. This typically
results in better sample quality at inference time.

Update rule:
    θ_ema = decay * θ_ema + (1 - decay) * θ_model

Typical decay values: 0.9999 or 0.999
"""

import copy
from typing import Optional, Iterable

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Usage:
        >>> model = UNet(...)
        >>> ema = EMA(model, decay=0.9999)
        >>> 
        >>> for batch in dataloader:
        ...     loss = compute_loss(model, batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     ema.update()  # Update EMA parameters
        >>> 
        >>> # For sampling, use EMA parameters
        >>> ema.apply_shadow()
        >>> samples = sample(model)
        >>> ema.restore()  # Restore training parameters
    
    Args:
        model: Model whose parameters to track
        decay: EMA decay rate (higher = slower update)
        warmup_steps: Number of steps before starting EMA (default: 0)
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
    ):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        
        # Shadow parameters (EMA of training parameters)
        self.shadow = {}
        
        # Backup of training parameters (for restore)
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def get_decay(self) -> float:
        """
        Get the current decay rate.
        
        During warmup, use a smaller decay to allow faster adaptation.
        
        Returns:
            Current decay rate
        """
        if self.step < self.warmup_steps:
            # Linear warmup of decay
            return min(self.decay, (1 + self.step) / (10 + self.step))
        return self.decay
    
    def update(self):
        """
        Update EMA parameters.
        
        Call this after each optimizer step.
        """
        decay = self.get_decay()
        self.step += 1
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # θ_ema = decay * θ_ema + (1 - decay) * θ_model
                self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_shadow(self):
        """
        Replace model parameters with EMA parameters.
        
        Call this before sampling. Remember to call restore() afterward
        if you want to continue training.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # Backup current parameters
                self.backup[name] = param.data.clone()
                # Replace with EMA parameters
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """
        Restore model parameters from backup.
        
        Call this after sampling to continue training with original parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        
        # Clear backup
        self.backup = {}
    
    def state_dict(self) -> dict:
        """
        Get state dict for checkpointing.
        
        Returns:
            Dictionary containing EMA state
        """
        return {
            'shadow': self.shadow,
            'step': self.step,
            'decay': self.decay,
        }
    
    def load_state_dict(self, state_dict: dict):
        """
        Load state dict from checkpoint.
        
        Args:
            state_dict: State dict to load
        """
        self.shadow = state_dict['shadow']
        self.step = state_dict['step']
        self.decay = state_dict.get('decay', self.decay)
    
    def to(self, device: torch.device) -> 'EMA':
        """
        Move EMA parameters to device.
        
        Args:
            device: Target device
        
        Returns:
            self for chaining
        """
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(device)
        return self
