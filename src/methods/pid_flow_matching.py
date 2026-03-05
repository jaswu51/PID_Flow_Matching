"""
PID Flow Matching (PID-FM)

Injects PID (Proportional-Integral-Derivative) control theory into Flow
Matching training and sampling.

Key idea:
- Training: perturb the ideal interpolation x_t and teach the model to output
  a corrected velocity that compensates for the perturbation via PID-weighted
  correction terms.
- Sampling: run a closed-loop PID controller that accumulates trajectory error
  history and feeds it back to adjust the predicted velocity in real time.

Reference framework:
  Standard FM target:  v_target = x_1 - x_0
  PID-FM target:       v_target = (x_1 - x_0) - (Kp * e_P + Ki * e_I + Kd * e_D)

where e_P is instantaneous perturbation, e_I simulates accumulated bias, and
e_D simulates sudden perturbation change.
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class PIDFlowMatching(BaseMethod):
    """
    PID-enhanced Flow Matching.

    Compared to standard FlowMatching, this method:
    1. Injects structured perturbations during training (P/I/D signals)
    2. Teaches the model to output PID-corrected velocity
    3. Uses closed-loop PID feedback during sampling
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_steps: int = 50,
        t_epsilon: float = 1e-5,
        # Timestep sampling schedule
        t_schedule: str = "uniform",       # "uniform" or "logit_normal"
        logit_normal_mean: float = 0.0,
        logit_normal_std: float = 1.0,
        # PID gains (training)
        kp: float = 0.5,
        ki: float = 0.1,
        kd: float = 0.05,
        # Perturbation magnitudes (training)
        sigma_p: float = 0.02,
        sigma_i: float = 0.01,
        sigma_d: float = 0.005,
        # PID gains (sampling) — can differ from training
        kp_sample: Optional[float] = None,
        ki_sample: Optional[float] = None,
        kd_sample: Optional[float] = None,
        # Whether to use PID perturbation during training
        pid_training: bool = True,
        # Whether to use closed-loop PID during sampling
        pid_sampling: bool = True,
    ):
        super().__init__(model, device)
        self.num_steps = int(num_steps)
        self.t_epsilon = float(t_epsilon)
        self.t_schedule = t_schedule
        self.logit_normal_mean = float(logit_normal_mean)
        self.logit_normal_std = float(logit_normal_std)

        # Training PID
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.sigma_p = float(sigma_p)
        self.sigma_i = float(sigma_i)
        self.sigma_d = float(sigma_d)

        # Sampling PID (default to training gains)
        self.kp_sample = float(kp_sample if kp_sample is not None else kp)
        self.ki_sample = float(ki_sample if ki_sample is not None else ki)
        self.kd_sample = float(kd_sample if kd_sample is not None else kd)

        self.pid_training = pid_training
        self.pid_sampling = pid_sampling

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample timestep t according to the configured schedule.

        "uniform":      t ~ U(0, 1)
        "logit_normal": t = sigmoid(mu + sigma * N(0,1))
                        Concentrates training away from high-variance endpoints.
        """
        if self.t_schedule == "logit_normal":
            eps = torch.randn(batch_size, device=device)
            t = torch.sigmoid(self.logit_normal_mean + self.logit_normal_std * eps)
        else:
            t = torch.rand(batch_size, device=device)
        if self.t_epsilon > 0:
            t = t.clamp(self.t_epsilon, 1.0 - self.t_epsilon)
        return t

    @staticmethod
    def _reshape_t(t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        return t.view(t.shape[0], *([1] * (len(x_shape) - 1)))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(
        self, x_1: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        PID-enhanced Flow Matching loss.

        When pid_training=True:
          1. Compute ideal x_t on the linear bridge
          2. Generate P, I, D perturbation signals
          3. Feed perturbed x_t to model
          4. Target = nominal velocity - PID correction
        When pid_training=False:
          Falls back to standard Flow Matching loss (for ablation).
        """
        batch_size = x_1.shape[0]
        x_0 = torch.randn_like(x_1)
        t = self._sample_t(batch_size, x_1.device)
        t_view = self._reshape_t(t, x_1.shape)

        # Ideal interpolation and nominal velocity
        x_t_ideal = (1.0 - t_view) * x_0 + t_view * x_1
        v_nominal = x_1 - x_0

        if not self.pid_training:
            # Standard FM
            v_pred = self.model(x_t_ideal, t)
            loss = F.mse_loss(v_pred, v_nominal)
            return loss, {"loss": loss.item(), "mse": loss.item()}

        # --- PID perturbation signals ---

        # P (proportional): instantaneous random offset
        # Scale perturbation by sqrt(t*(1-t)) so it's small near endpoints
        scale = torch.sqrt(t_view * (1.0 - t_view) + 1e-8)
        delta_p = torch.randn_like(x_t_ideal) * self.sigma_p * scale

        # I (integral): low-frequency bias (smooth, spatially correlated)
        # Use pooled noise to simulate accumulated drift
        delta_i_raw = torch.randn_like(x_t_ideal)
        if x_t_ideal.dim() == 4 and x_t_ideal.shape[-1] >= 4:
            # Smooth it: downsample then upsample
            h, w = x_t_ideal.shape[2], x_t_ideal.shape[3]
            delta_i_smooth = F.interpolate(
                F.avg_pool2d(delta_i_raw, kernel_size=4, stride=4),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
        else:
            delta_i_smooth = delta_i_raw
        # Scale by t to simulate accumulation over time
        delta_i = delta_i_smooth * self.sigma_i * t_view

        # D (derivative): high-frequency perturbation (sudden change)
        delta_d = torch.randn_like(x_t_ideal) * self.sigma_d

        # Perturbed input
        x_t_perturbed = x_t_ideal + delta_p + delta_i + delta_d

        # PID-corrected target velocity
        correction = (
            self.kp * delta_p
            + self.ki * delta_i
            + self.kd * delta_d
        )
        v_target = v_nominal - correction

        # Model prediction on perturbed input
        v_pred = self.model(x_t_perturbed, t)
        loss = F.mse_loss(v_pred, v_target)

        # Metrics
        with torch.no_grad():
            correction_norm = correction.flatten(1).norm(dim=1).mean().item()
            perturbation_norm = (delta_p + delta_i + delta_d).flatten(1).norm(dim=1).mean().item()

        metrics = {
            "loss": loss.item(),
            "mse": loss.item(),
            "correction_norm": correction_norm,
            "perturbation_norm": perturbation_norm,
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        num_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        ODE sampling with optional closed-loop PID correction.

        When pid_sampling=True:
          Maintains integral_error and prev_error buffers across steps.
          At each step, estimates trajectory error and applies PID correction
          to the predicted velocity before the Euler update.
        When pid_sampling=False:
          Standard Euler integration (for ablation).
        """
        self.eval_mode()
        device = self.device
        x = torch.randn((batch_size, *image_shape), device=device)

        steps = num_steps or self.num_steps
        t_vals = torch.linspace(0.0, 1.0, steps + 1, device=device)
        dt = t_vals[1] - t_vals[0]

        if not self.pid_sampling:
            # Standard Euler
            for i in range(steps):
                t = torch.full((batch_size,), t_vals[i], device=device)
                v = self.model(x, t)
                x = x + v * dt
            return x

        # --- Closed-loop PID sampling ---
        x_0_init = x.clone()  # remember initial noise for reference trajectory
        integral_error = torch.zeros_like(x)
        prev_error = torch.zeros_like(x)

        for i in range(steps):
            t_cur = t_vals[i]
            t_next = t_vals[i + 1]
            t = torch.full((batch_size,), t_cur, device=device)

            # Model prediction
            v_pred = self.model(x, t)

            # Estimate where we "should" be on the ideal straight line:
            # If we went from x_0_init at t=0 with constant velocity v_pred,
            # the reference at t_cur is x_0_init + t_cur * v_pred_accumulated.
            # Simpler proxy: use the model's own velocity to estimate x_1,
            # then compute reference as (1-t)*x_0 + t*x_1_est.
            x_1_est = x + (1.0 - t_cur) * v_pred  # one-step estimate of x_1
            x_ref = (1.0 - t_cur) * x_0_init + t_cur * x_1_est

            # Current error
            current_error = x - x_ref

            # PID terms
            integral_error = integral_error + current_error * dt.item()
            derivative_error = (current_error - prev_error) / max(dt.item(), 1e-8)

            pid_correction = (
                self.kp_sample * current_error
                + self.ki_sample * integral_error
                + self.kd_sample * derivative_error
            )

            # Corrected velocity
            v_corrected = v_pred - pid_correction

            # Euler step
            x = x + v_corrected * dt

            prev_error = current_error

        return x

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to(self, device: torch.device) -> "PIDFlowMatching":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_steps"] = self.num_steps
        state["t_epsilon"] = self.t_epsilon
        state["t_schedule"] = self.t_schedule
        state["logit_normal_mean"] = self.logit_normal_mean
        state["logit_normal_std"] = self.logit_normal_std
        state["kp"] = self.kp
        state["ki"] = self.ki
        state["kd"] = self.kd
        state["sigma_p"] = self.sigma_p
        state["sigma_i"] = self.sigma_i
        state["sigma_d"] = self.sigma_d
        return state

    @classmethod
    def from_config(
        cls, model: nn.Module, config: dict, device: torch.device
    ) -> "PIDFlowMatching":
        pid_config = config.get("pid_flow_matching", config.get("flow_matching", config))
        method = cls(
            model=model,
            device=device,
            num_steps=pid_config.get("num_steps", 50),
            t_epsilon=pid_config.get("t_epsilon", 1e-5),
            t_schedule=pid_config.get("t_schedule", "uniform"),
            logit_normal_mean=pid_config.get("logit_normal_mean", 0.0),
            logit_normal_std=pid_config.get("logit_normal_std", 1.0),
            kp=pid_config.get("kp", 0.5),
            ki=pid_config.get("ki", 0.1),
            kd=pid_config.get("kd", 0.05),
            sigma_p=pid_config.get("sigma_p", 0.02),
            sigma_i=pid_config.get("sigma_i", 0.01),
            sigma_d=pid_config.get("sigma_d", 0.005),
            kp_sample=pid_config.get("kp_sample", None),
            ki_sample=pid_config.get("ki_sample", None),
            kd_sample=pid_config.get("kd_sample", None),
            pid_training=pid_config.get("pid_training", True),
            pid_sampling=pid_config.get("pid_sampling", True),
        )
        return method.to(device)
