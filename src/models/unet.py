"""
U-Net Architecture for Diffusion Models

In this file, you should implements a U-Net architecture suitable for DDPM.

Architecture Overview:
    Input: (batch_size, channels, H, W), timestep
    
    Encoder (Downsampling path)

    Middle
    
    Decoder (Upsampling path)
    
    Output: (batch_size, channels, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .blocks import (
    TimestepEmbedding,
    ResBlock,
    AttentionBlock,
    Downsample,
    Upsample,
    GroupNorm32,
)


class UNet(nn.Module):
    """
    TODO: design your own U-Net architecture for diffusion models.

    Args:
        in_channels: Number of input image channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        base_channels: Base channel count (multiplied by channel_mult at each level)
        channel_mult: Tuple of channel multipliers for each resolution level
                     e.g., (1, 2, 4, 8) means channels are [C, 2C, 4C, 8C]
        num_res_blocks: Number of residual blocks per resolution level
        attention_resolutions: Resolutions at which to apply self-attention
                              e.g., [16, 8] applies attention at 16x16 and 8x8
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_scale_shift_norm: Whether to use FiLM conditioning in ResBlocks
    
    Example:
        >>> model = UNet(
        ...     in_channels=3,
        ...     out_channels=3, 
        ...     base_channels=128,
        ...     channel_mult=(1, 2, 2, 4),
        ...     num_res_blocks=2,
        ...     attention_resolutions=[16, 8],
        ... )
        >>> x = torch.randn(4, 3, 64, 64)
        >>> t = torch.randint(0, 1000, (4,))
        >>> out = model(x, t)
        >>> out.shape
        torch.Size([4, 3, 64, 64])
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        num_heads: int = 4,
        dropout: float = 0.1,
        use_scale_shift_norm: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_scale_shift_norm = use_scale_shift_norm
        
        time_embed_dim = base_channels * 4
        self.time_embed = TimestepEmbedding(time_embed_dim, hidden_dim=time_embed_dim)

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        self.up_blocks = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        self.mid_block1 = ResBlock(
            base_channels * channel_mult[-1],
            base_channels * channel_mult[-1],
            time_embed_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.mid_attn = AttentionBlock(base_channels * channel_mult[-1], num_heads=num_heads)
        self.mid_block2 = ResBlock(
            base_channels * channel_mult[-1],
            base_channels * channel_mult[-1],
            time_embed_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self.out_norm = GroupNorm32(32, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

        # Build encoder
        resolution = 64
        ch = base_channels
        down_channels = []
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock(
                        ch,
                        out_ch,
                        time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                if resolution in attention_resolutions:
                    self.down_attn.append(AttentionBlock(out_ch, num_heads=num_heads))
                else:
                    self.down_attn.append(nn.Identity())
                ch = out_ch
                down_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.downsamples.append(Downsample(ch))
                resolution //= 2

        # Build decoder
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = down_channels.pop()
                self.up_blocks.append(
                    ResBlock(
                        ch + skip_ch,
                        out_ch,
                        time_embed_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                if resolution in attention_resolutions:
                    self.up_attn.append(AttentionBlock(out_ch, num_heads=num_heads))
                else:
                    self.up_attn.append(nn.Identity())
                ch = out_ch
            if level != 0:
                self.upsamples.append(Upsample(ch))
                resolution *= 2
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement the forward pass of the unet
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
               This is typically the noisy image x_t
            t: Timestep tensor of shape (batch_size,)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        time_emb = self.time_embed(t)

        h = self.input_conv(x)
        hs = []

        down_idx = 0
        downsample_idx = 0
        for level, _ in enumerate(self.channel_mult):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[down_idx](h, time_emb)
                h = self.down_attn[down_idx](h)
                hs.append(h)
                down_idx += 1
            if level != len(self.channel_mult) - 1:
                h = self.downsamples[downsample_idx](h)
                downsample_idx += 1

        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)

        up_idx = 0
        upsample_idx = 0
        for level, _ in reversed(list(enumerate(self.channel_mult))):
            for _ in range(self.num_res_blocks):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[up_idx](h, time_emb)
                h = self.up_attn[up_idx](h)
                up_idx += 1
            if level != 0:
                h = self.upsamples[upsample_idx](h)
                upsample_idx += 1

        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        return h


def create_model_from_config(config: dict) -> UNet:
    """
    Factory function to create a UNet from a configuration dictionary.
    
    Args:
        config: Dictionary containing model configuration
                Expected to have a 'model' key with the relevant parameters
    
    Returns:
        Instantiated UNet model
    """
    model_config = config['model']
    data_config = config['data']
    
    return UNet(
        in_channels=data_config['channels'],
        out_channels=data_config['channels'],
        base_channels=model_config['base_channels'],
        channel_mult=tuple(model_config['channel_mult']),
        num_res_blocks=model_config['num_res_blocks'],
        attention_resolutions=model_config['attention_resolutions'],
        num_heads=model_config['num_heads'],
        dropout=model_config['dropout'],
        use_scale_shift_norm=model_config['use_scale_shift_norm'],
    )


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the model
    print("Testing UNet...")
    
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        num_heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 64, 64)
    t = torch.rand(batch_size)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Forward pass successful!")
