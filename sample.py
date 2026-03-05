"""
Sampling Script for DDPM (Denoising Diffusion Probabilistic Models)

Generate samples from a trained model. By default, saves individual images to avoid
memory issues with large sample counts. Use --grid to generate a single grid image.

Usage:
    # Sample from DDPM (saves individual images to ./samples/)
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64

    # With custom number of sampling steps
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_steps 500

    # Generate a grid image instead of individual images
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --num_samples 64 --grid

    # Save individual images to custom directory
    python sample.py --checkpoint checkpoints/ddpm_final.pt --method ddpm --output_dir my_samples

What you need to implement:
- Incorporate your sampling scheme to this pipeline
- Save generated samples as images for logging
"""

import os
import sys
import argparse
from datetime import datetime

import yaml
import torch
from tqdm import tqdm

from src.models import create_model_from_config
from src.models.dit import create_dit_from_config
from src.data import save_image, unnormalize
from src.methods import DDPM, FlowMatching, RectifiedFlow, PIDFlowMatching, MeanFlow, OnPolicyFlowMatching
from src.utils import EMA


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint and return model, config, and EMA."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Create model (handle both UNet and DiT)
    model_type = config.get('model', {}).get('type', 'unet')
    if model_type == 'dit':
        model = create_dit_from_config(config).to(device)
    else:
        model = create_model_from_config(config).to(device)
    model.load_state_dict(checkpoint['model'])

    # Create EMA and load
    ema = EMA(model, decay=config['training']['ema_decay'])
    ema.load_state_dict(checkpoint['ema'])

    return model, config, ema


def save_samples(
    samples: torch.Tensor,
    save_path: str,
    num_samples: int,
    nrow: int = None,
) -> None:
    """
    TODO: save generated samples as images.

    Args:
        samples: Generated samples tensor with shape (num_samples, C, H, W).
        save_path: File path to save the image grid.
        num_samples: Number of samples, used to calculate grid layout.
    """

    # Unnormalize from [-1, 1] to [0, 1] for saving
    samples = unnormalize(samples.detach().cpu())
    if nrow is None:
        nrow = int(num_samples ** 0.5)
    nrow = max(1, nrow)
    save_image(samples, save_path, nrow=nrow)


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ddpm', 'flow_matching', 'rectified_flow', 'reflow', 'pid_flow_matching', 'meanflow', 'on_policy_fm'],
                       help='Method used for training')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='samples',
                       help='Directory to save individual images (default: samples)')
    parser.add_argument('--grid', action='store_true',
                       help='Save as grid image instead of individual images')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for grid (only used with --grid, default: samples_<timestamp>.png)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    # Sampling arguments
    parser.add_argument('--num_steps', type=int, default=None,
                       help='Number of sampling steps (default: from config)')
    parser.add_argument('--sampler', type=str, default=None,
                       choices=['ddpm', 'ddim', 'flow_matching'],
                       help='Sampling algorithm (default: from config if available)')
    parser.add_argument('--eta', type=float, default=0.0,
                       help='DDIM eta (0.0 for deterministic sampling)')
    
    # PID sampling gain overrides (pid_flow_matching only)
    parser.add_argument('--kp_sample', type=float, default=None,
                       help='PID kp for sampling (overrides checkpoint config; None = use training kp)')
    parser.add_argument('--ki_sample', type=float, default=None,
                       help='PID ki for sampling (overrides checkpoint config; None = use training ki)')
    parser.add_argument('--kd_sample', type=float, default=None,
                       help='PID kd for sampling (overrides checkpoint config; None = use training kd)')

    # Other options
    parser.add_argument('--no_ema', action='store_true',
                       help='Use training weights instead of EMA weights')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, config, ema = load_checkpoint(args.checkpoint, device)

    # Apply PID sampling gain overrides before instantiation
    if args.method == 'pid_flow_matching':
        pid_cfg = config.setdefault('pid_flow_matching', {})
        if args.kp_sample is not None:
            pid_cfg['kp_sample'] = args.kp_sample
        elif 'kp_sample' not in pid_cfg:
            pid_cfg['kp_sample'] = None  # will default to kp in __init__
        if args.ki_sample is not None:
            pid_cfg['ki_sample'] = args.ki_sample
        elif 'ki_sample' not in pid_cfg:
            pid_cfg['ki_sample'] = None
        if args.kd_sample is not None:
            pid_cfg['kd_sample'] = args.kd_sample
        elif 'kd_sample' not in pid_cfg:
            pid_cfg['kd_sample'] = None
        if args.kp_sample is not None or args.ki_sample is not None or args.kd_sample is not None:
            print(f"PID sampling gains overridden: kp_s={pid_cfg.get('kp_sample')}, "
                  f"ki_s={pid_cfg.get('ki_sample')}, kd_s={pid_cfg.get('kd_sample')}")

    # Create method
    if args.method == 'ddpm':
        method = DDPM.from_config(model, config, device)
    elif args.method == 'flow_matching':
        method = FlowMatching.from_config(model, config, device)
    elif args.method in ('rectified_flow', 'reflow'):
        method = RectifiedFlow.from_config(model, config, device)
    elif args.method == 'pid_flow_matching':
        method = PIDFlowMatching.from_config(model, config, device)
    elif args.method == 'meanflow':
        method = MeanFlow.from_config(model, config, device)
    elif args.method == 'on_policy_fm':
        method = OnPolicyFlowMatching.from_config(model, config, device)
    else:
        raise ValueError(
            f"Unknown method: {args.method}. "
            f"Supported: 'ddpm', 'flow_matching', 'rectified_flow', 'reflow', 'pid_flow_matching', 'meanflow', 'on_policy_fm'."
        )
    
    # Apply EMA weights
    if not args.no_ema:
        print("Using EMA weights")
        ema.apply_shadow()
    else:
        print("Using training weights (no EMA)")
    
    method.eval_mode()
    
    # Image shape
    data_config = config['data']
    image_shape = (data_config['channels'], data_config['image_size'], data_config['image_size'])
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")

    all_samples = []
    remaining = args.num_samples
    sample_idx = 0

    # Create output directory if saving individual images
    if not args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")
        while remaining > 0:
            batch_size = min(args.batch_size, remaining)

            num_steps = args.num_steps or config['sampling']['num_steps']
            sampler = args.sampler or config.get('sampling', {}).get('sampler', None)

            samples = method.sample(
                batch_size=batch_size,
                image_shape=image_shape,
                num_steps=num_steps,
                sampler=sampler,
                eta=args.eta,
            )

            # Save individual images immediately or collect for grid
            if args.grid:
                all_samples.append(samples)
            else:
                for i in range(samples.shape[0]):
                    img_path = os.path.join(args.output_dir, f"{sample_idx:06d}.png")
                    save_samples(samples[i:i+1], img_path, 1)
                    sample_idx += 1

            remaining -= batch_size
            pbar.update(batch_size)

        pbar.close()

    # Save samples
    if args.grid:
        # Concatenate all samples for grid
        all_samples = torch.cat(all_samples, dim=0)[:args.num_samples]

        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output = f"samples_{timestamp}.png"

        save_samples(all_samples, args.output, args.num_samples)
        print(f"Saved grid to {args.output}")
    else:
        print(f"Saved {args.num_samples} individual images to {args.output_dir}")

    # Restore EMA if applied
    if not args.no_ema:
        ema.restore()


if __name__ == '__main__':
    main()
