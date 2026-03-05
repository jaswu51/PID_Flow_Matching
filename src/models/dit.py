"""
DiT (Diffusion Transformer) Architecture

Implements the Diffusion Transformer from Peebles & Xie (2023):
"Scalable Diffusion Models with Transformers"

Key components:
- Patchification: image -> sequence of patch tokens
- Transformer blocks with adaLN-Zero conditioning
- Unpatchify: sequence of tokens -> predicted image/velocity

Reference: https://github.com/facebookresearch/DiT
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# =============================================================================
# Positional Embeddings
# =============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for scalar timestep t."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device) / half_dim
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepEmbedder(nn.Module):
    """MLP to embed scalar timestep into a vector."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.sinusoidal = SinusoidalPositionalEmbedding(frequency_embedding_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.sinusoidal(t)
        return self.mlp(t_emb)


# =============================================================================
# Patch Embedding
# =============================================================================

class PatchEmbed(nn.Module):
    """2D image -> patch embedding via Conv2d."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


# =============================================================================
# adaLN-Zero Modulation
# =============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive layer norm modulation: x * (1 + scale) + shift."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# =============================================================================
# DiT Block
# =============================================================================

class DiTBlock(nn.Module):
    """
    Transformer block with adaLN-Zero conditioning.

    Each block has:
    1. LayerNorm -> Self-Attention (modulated by adaLN)
    2. LayerNorm -> MLP (modulated by adaLN)

    The modulation parameters (shift, scale, gate) are predicted from
    the conditioning signal c via a shared linear layer.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, mlp_hidden)

        # adaLN-Zero: predict 6 modulation parameters from c
        # (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Zero-init the modulation output so blocks act as identity at init
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token sequence
            c: (B, D) conditioning vector
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )

        # Self-attention with adaLN modulation
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )

        # MLP with adaLN modulation
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

        return x


class Attention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention (uses Flash Attention if available)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(x)


class Mlp(nn.Module):
    """MLP with GELU activation (as in ViT/DiT)."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# =============================================================================
# Final Layer
# =============================================================================

class FinalLayer(nn.Module):
    """Final layer: adaLN-Zero modulated LayerNorm + linear projection."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )
        # Zero-init
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# =============================================================================
# DiT Model
# =============================================================================

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT).

    Takes an image and a timestep, outputs the predicted velocity (or noise).
    Uses patchification + Transformer blocks + adaLN-Zero conditioning.

    Args:
        img_size: Input image size (assumes square).
        patch_size: Patch size for patchification.
        in_channels: Number of input image channels.
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim = hidden_size * mlp_ratio.
        dual_time: If True, adds a second timestep embedder for Mean Flow (r, t).
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dual_time: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_patches = (img_size // patch_size) ** 2
        self.dual_time = dual_time

        # Patch embedding
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, hidden_size)

        # Timestep embedding (t)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Second timestep embedder for r (Mean Flow only)
        if dual_time:
            self.r_embedder = TimestepEmbedder(hidden_size)

        # Learnable positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )

        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following DiT paper."""
        # Initialize patch embedding like a linear layer
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.x_embedder.proj.bias)

        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Initialize transformer blocks
        for block in self.blocks:
            # Attention: Xavier uniform for qkv and proj
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.zeros_(block.attn.qkv.bias)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.zeros_(block.attn.proj.bias)
            # MLP
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            nn.init.zeros_(block.mlp.fc2.bias)

        # Timestep embedder MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.t_embedder.mlp[0].bias)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.t_embedder.mlp[2].bias)

        # r embedder (Mean Flow)
        if self.dual_time:
            nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
            nn.init.zeros_(self.r_embedder.mlp[0].bias)
            nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)
            nn.init.zeros_(self.r_embedder.mlp[2].bias)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens back to image.
        x: (B, num_patches, patch_size**2 * C)
        returns: (B, C, H, W)
        """
        p = self.patch_size
        h = w = self.img_size // p
        c = self.out_channels
        x = x.reshape(-1, h, w, p, p, c)
        x = rearrange(x, "b h w p1 p2 c -> b c (h p1) (w p2)")
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor, r: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C, H, W) input image (or noisy image)
            t: (B,) timestep values
            r: (B,) reference timestep for Mean Flow (optional, requires dual_time=True)

        Returns:
            (B, C, H, W) predicted velocity / average velocity
        """
        # Patchify + position embedding
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D)

        # Timestep conditioning
        c = self.t_embedder(t)  # (B, D)
        if self.dual_time and r is not None:
            c = c + self.r_embedder(r)  # add r embedding (Mean Flow)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Final layer + unpatchify
        x = self.final_layer(x, c)  # (B, N, p*p*C)
        x = self.unpatchify(x)  # (B, C, H, W)

        return x


# =============================================================================
# Model Configurations
# =============================================================================

DiT_configs = {
    "DiT-S": {"hidden_size": 384, "depth": 12, "num_heads": 6},
    "DiT-B": {"hidden_size": 768, "depth": 12, "num_heads": 12},
    "DiT-L": {"hidden_size": 1024, "depth": 24, "num_heads": 16},
    "DiT-XL": {"hidden_size": 1152, "depth": 28, "num_heads": 16},
}


def create_dit_from_config(config: dict) -> DiT:
    """
    Factory function to create a DiT from a configuration dictionary.

    Supports two modes:
    1. Named variant: config['model']['dit_variant'] = 'DiT-S' (uses preset configs)
    2. Custom: specify hidden_size, depth, num_heads directly in config['model']
    """
    model_config = config["model"]
    data_config = config["data"]

    # Check for named variant
    variant = model_config.get("dit_variant", None)
    if variant and variant in DiT_configs:
        preset = DiT_configs[variant]
        hidden_size = preset["hidden_size"]
        depth = preset["depth"]
        num_heads = preset["num_heads"]
    else:
        hidden_size = model_config.get("hidden_size", 384)
        depth = model_config.get("depth", 12)
        num_heads = model_config.get("num_heads", 6)

    return DiT(
        img_size=data_config["image_size"],
        patch_size=model_config.get("patch_size", 4),
        in_channels=data_config["channels"],
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=model_config.get("mlp_ratio", 4.0),
        dual_time=model_config.get("dual_time", False),
    )
