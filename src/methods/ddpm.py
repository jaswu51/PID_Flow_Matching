"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        parameterization: str = "eps",
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.parameterization = parameterization

        betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], dim=0
        )

        # Register buffers so they move with the module/device
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(self.posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        if self.parameterization not in {"eps", "x0", "v"}:
            raise ValueError(
                f"Unsupported parameterization '{self.parameterization}'. "
                "Use one of: 'eps', 'x0', 'v'."
            )

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================
    
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that
    
    # =========================================================================
    # Forward process
    # =========================================================================

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus * noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TODO: Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """

        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)
        noise = torch.randn_like(x_0)

        x_t = self.forward_process(x_0, t, noise)

        if self.parameterization == "eps":
            pred = self.model(x_t, t)
            target = noise
        elif self.parameterization == "x0":
            pred = self.model(x_t, t)
            target = x_0
        else:  # "v"
            pred = self.model(x_t, t)
            sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
            sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
            target = sqrt_alpha_cumprod * noise - sqrt_one_minus * x_0

        loss = F.mse_loss(pred, target)
        metrics = {"loss": loss.item(), "mse": loss.item()}
        return loss, metrics

    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process_ddpm(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments
        
        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        pred = self.model(x_t, t)
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        if self.parameterization == "eps":
            pred_noise = pred
            x_0_pred = (x_t - sqrt_one_minus * pred_noise) / sqrt_alpha_cumprod
        elif self.parameterization == "x0":
            x_0_pred = pred
        else:  # "v"
            v = pred
            x_0_pred = sqrt_alpha_cumprod * x_t - sqrt_one_minus * v

        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_pred
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if torch.all(t == 0):
            return posterior_mean

        noise = torch.randn_like(x_t)
        log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean + torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def reverse_process_ddim(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        One step of DDIM reverse process.
        """
        pred = self.model(x_t, t)
        sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)

        if self.parameterization == "eps":
            eps_pred = pred
            x_0_pred = (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha_cumprod
        elif self.parameterization == "x0":
            x_0_pred = pred
            eps_pred = (x_t - sqrt_alpha_cumprod * x_0_pred) / sqrt_one_minus
        else:  # "v"
            v = pred
            x_0_pred = sqrt_alpha_cumprod * x_t - sqrt_one_minus * v
            eps_pred = sqrt_one_minus * x_t + sqrt_alpha_cumprod * v

        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        alpha_bar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x_t.shape)

        sigma_t = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
        )
        noise = torch.randn_like(x_t)

        x_prev = (
            torch.sqrt(alpha_bar_prev) * x_0_pred
            + torch.sqrt(1.0 - alpha_bar_prev - sigma_t**2) * eps_pred
            + sigma_t * noise
        )
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        sampler: str = "ddpm",
        eta: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Implement DDPM/DDIM sampling loop.
        """
        self.eval_mode()
        device = self.device
        x = torch.randn((batch_size, *image_shape), device=device)

        # If num_steps is not provided, use the default num_timesteps
        num_steps = num_steps or self.num_timesteps

        # Create a linear sequence of timesteps to sample from
        # e.g., if num_steps=100 and self.num_timesteps=1000, indices will be 999, 989, 979...
        indices = torch.linspace(self.num_timesteps - 1, 0, steps=num_steps).long()

        if sampler == "ddpm":
            for i in tqdm(indices, desc=f"Sampling ({num_steps} steps)", leave=False):
                t = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x = self.reverse_process_ddpm(x, t)
            return x

        if sampler != "ddim":
            raise ValueError(f"Unknown sampler: {sampler}. Use 'ddpm' or 'ddim'.")

        # DDIM sampling
        for idx, i in enumerate(tqdm(indices, desc=f"DDIM Sampling ({num_steps} steps)", leave=False)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            if idx == len(indices) - 1:
                pred = self.model(x, t)
                sqrt_alpha_cumprod = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
                sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

                if self.parameterization == "eps":
                    x = (x - sqrt_one_minus * pred) / sqrt_alpha_cumprod
                elif self.parameterization == "x0":
                    x = pred
                else:  # "v"
                    x = sqrt_alpha_cumprod * x - sqrt_one_minus * pred
                x = torch.clamp(x, -1.0, 1.0)
                continue

            prev_i = indices[idx + 1]
            t_prev = torch.full((batch_size,), prev_i, device=device, dtype=torch.long)
            x = self.reverse_process_ddim(x, t, t_prev, eta=eta)

        return x

    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        state["beta_start"] = self.beta_start
        state["beta_end"] = self.beta_end
        state["parameterization"] = self.parameterization
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        method = cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            parameterization=ddpm_config.get("parameterization", "eps"),
        )
        return method.to(device)

    # =========================================================================
    # Helper functions
    # =========================================================================

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """
        Extract coefficients at timesteps t and reshape for broadcasting.
        """
        out = a.gather(0, t)
        view_shape = (t.shape[0],) + (1,) * (len(x_shape) - 1)
        return out.view(view_shape)
