"""
Flow Matching (FM)

This module implements a simple Flow Matching method with a linear interpolation
bridge and an ODE-based sampler.
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class FlowMatching(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_steps: int = 50,
        t_epsilon: float = 1e-5,
    ):
        super().__init__(model, device)
        self.num_steps = int(num_steps)
        self.t_epsilon = float(t_epsilon)

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        t = torch.rand(batch_size, device=device)
        if self.t_epsilon > 0:
            t = t * (1.0 - 2.0 * self.t_epsilon) + self.t_epsilon
        return t

    @staticmethod
    def _reshape_t(t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        return t.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    def compute_loss(self, x_1: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Flow Matching loss with a linear interpolation bridge:
        x_t = (1 - t) * x_0 + t * x_1, and target velocity v = x_1 - x_0.
        """
        batch_size = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = self._sample_t(batch_size, x_1.device)
        t_view = self._reshape_t(t, x_1.shape)

        x_t = (1.0 - t_view) * x_0 + t_view * x_1
        v_target = x_1 - x_0

        v_pred = self.model(x_t, t)
        loss = F.mse_loss(v_pred, v_target)
        metrics = {"loss": loss.item(), "mse": loss.item()}
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        ODE sampling with Euler integration from t=0 to t=1.
        """
        self.eval_mode()
        device = self.device
        x = torch.randn((batch_size, *image_shape), device=device)

        steps = num_steps or self.num_steps
        t_vals = torch.linspace(0.0, 1.0, steps + 1, device=device)
        dt = t_vals[1] - t_vals[0]

        for i in range(steps):
            t = torch.full((batch_size,), t_vals[i], device=device)
            v = self.model(x, t)
            x = x + v * dt

        return x

    def to(self, device: torch.device) -> "FlowMatching":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_steps"] = self.num_steps
        state["t_epsilon"] = self.t_epsilon
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "FlowMatching":
        fm_config = config.get("flow_matching", config)
        method = cls(
            model=model,
            device=device,
            num_steps=fm_config.get("num_steps", 50),
            t_epsilon=fm_config.get("t_epsilon", 1e-5),
        )
        return method.to(device)
