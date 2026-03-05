"""
Methods module for PID Flow Matching.

Implementations:
- BaseMethod: abstract base class
- FlowMatching: standard Flow Matching
- RectifiedFlow: Rectified Flow with logit-normal t-sampling support
- PIDFlowMatching: PID-enhanced Rectified Flow (this project's contribution)
- DDPM: Denoising Diffusion Probabilistic Models (legacy baseline)
"""

from .base import BaseMethod
from .ddpm import DDPM
from .flow_matching import FlowMatching
from .rectified_flow import RectifiedFlow
from .pid_flow_matching import PIDFlowMatching

__all__ = [
    'BaseMethod',
    'DDPM',
    'FlowMatching',
    'RectifiedFlow',
    'PIDFlowMatching',
]
