"""
U-Net Building Blocks for Diffusion Models

This module contains the fundamental components for building a U-Net architecture
suitable for diffusion models and flow matching:
- Sinusoidal positional embeddings for time conditioning
- Residual blocks with optional time conditioning (FiLM)
- Multi-head self-attention blocks
- Downsampling and upsampling layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Time Embeddings
# =============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t can be integer timesteps or continuous values.
        """
        device = t.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -math.log(self.max_period) *
            torch.arange(half_dim, device=device) / half_dim
        )

        args = t.float()[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class TimestepEmbedding(nn.Module):
    def __init__(self, time_embed_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * time_embed_dim

        self.sinusoidal = SinusoidalPositionalEmbedding(time_embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal(t)
        emb = self.mlp(emb)
        return emb


# =============================================================================
# Normalization Layers
# =============================================================================

class GroupNorm32(nn.GroupNorm):
    """
    Group normalization with float32 precision for stability.
    
    Diffusion models can be sensitive to numerical precision in normalization
    layers, so we cast to float32 before normalizing and back afterward.
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


# =============================================================================
# Residual Blocks
# =============================================================================

class ResBlock(nn.Module):
    """
    Residual block with time conditioning.
    
    This is the workhorse of the U-Net. Each block consists of:
    1. GroupNorm -> SiLU -> Conv
    2. Add time embedding (either add or scale+shift via FiLM)
    3. GroupNorm -> SiLU -> Dropout -> Conv
    4. Residual connection (with optional channel adjustment)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        time_embed_dim: Dimension of time embedding
        dropout: Dropout probability
        use_scale_shift_norm: If True, use FiLM conditioning (scale and shift)
                             If False, just add time embedding
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        
        # First convolution block
        self.norm1 = GroupNorm32(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        # If using scale_shift_norm, we need 2x channels (for scale and shift)
        time_out_dim = out_channels * 2 if use_scale_shift_norm else out_channels
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_out_dim),
        )
        
        # Second convolution block
        self.norm2 = GroupNorm32(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Residual connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            time_emb: Time embedding of shape (batch_size, time_embed_dim)
        
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        time_emb = time_emb[:, :, None, None]
        
        if self.use_scale_shift_norm:
            # FiLM conditioning: scale and shift
            scale, shift = torch.chunk(time_emb, 2, dim=1)
            h = self.norm2(h)
            h = h * (1 + scale) + shift
            h = F.silu(h)
        else:
            # Simple addition
            h = h + time_emb
            h = self.norm2(h)
            h = F.silu(h)
        
        # Second conv block
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Residual connection
        return h + self.skip_connection(x)


# =============================================================================
# Attention Blocks
# =============================================================================

class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block.
    
    Applies self-attention over spatial dimensions (height x width).
    Used at lower resolutions in the U-Net where the spatial dimensions
    are small enough to make attention computationally feasible.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads
    """
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, \
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        self.norm = GroupNorm32(32, channels)
        
        # QKV projection (combined for efficiency)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        
        # Output projection
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Scale factor for dot-product attention
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Output tensor of shape (batch_size, channels, height, width)
        """
        B, C, H, W = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h)
        qkv = rearrange(qkv, 'b (three heads head_dim) h w -> three b heads (h w) head_dim',
                       three=3, heads=self.num_heads, head_dim=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: softmax(Q @ K^T / sqrt(d)) @ V
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        # Reshape back to spatial dimensions
        out = rearrange(out, 'b heads (h w) head_dim -> b (heads head_dim) h w',
                       h=H, w=W, heads=self.num_heads, head_dim=self.head_dim)
        
        # Output projection and residual
        out = self.proj_out(out)
        
        return x + out


# =============================================================================
# Up/Downsampling Layers
# =============================================================================

class Downsample(nn.Module):
    """
    Downsampling layer that halves spatial dimensions.
    
    Uses strided convolution instead of pooling to allow the network
    to learn the downsampling operation.
    
    Args:
        channels: Number of input/output channels
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """
    Upsampling layer that doubles spatial dimensions.
    
    Uses nearest-neighbor upsampling followed by convolution.
    This is more stable than transposed convolution (avoids checkerboard artifacts).
    
    Args:
        channels: Number of input/output channels
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
