"""
Modal Configuration for CMU 10799 Diffusion Homework

Defines Modal environment and training functions for cloud GPU training.

See docs/QUICKSTART-MODAL.md for setup and usage instructions.

All parameters are read from config YAML files first, then overridden by command-line arguments.
"""

import modal

# =============================================================================
# Modal App Definition
# =============================================================================

# Create the Modal app
app = modal.App("cmu-10799-diffusion")

# Define the container image with all dependencies
# This mirrors the CPU-only environment (environments/environment-cpu.yml)
# but installs GPU-enabled PyTorch automatically on Modal's GPU machines
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
        "wandb>=0.15.0",
        "datasets>=2.0.0",  # For HuggingFace Hub dataset loading
        "torch-fidelity>=0.3.0",  # Comprehensive evaluation metrics
        "scikit-learn>=1.0.0",  # PCA for trajectory visualization
        "matplotlib>=3.5.0",  # Plotting
    )
    # Copy the local project directory into the image
    .add_local_dir(
        ".",
        "/root",
        ignore=[
            ".git",
            ".venv*",
            "venv",
            "__pycache__",
            "logs",
            "checkpoints",
            "data",
            "*.md",
            "docs",
            "environments",
            "notebooks",
        ],
    )
)

# Create a persistent volume for checkpoints and data
volume = modal.Volume.from_name("cmu-10799-diffusion-data", create_if_missing=True)

# =============================================================================
# Training Function
# =============================================================================

def _train_impl(
    method: str,
    config_path: str,
    resume_from: str,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    """
    Internal training implementation.

    Reads config from YAML file, applies command-line overrides.
    """
    import os
    import sys
    import yaml
    import tempfile
    import subprocess

    sys.path.insert(0, "/root")

    # Load config
    config_tag = method
    if config_path is None:
        config_path = f"/root/configs/{method}.yaml"
    else:
        config_path = f"/root/{config_path}"
        config_tag = os.path.splitext(os.path.basename(config_path))[0]

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get num_gpus from config
    config_device = config['infrastructure'].get('device', 'cuda')
    num_gpus = config['infrastructure'].get('num_gpus', 1 if config_device == 'cuda' else 0)
    if num_gpus is None:
        num_gpus = 1 if config_device == 'cuda' else 0

    # Read from_hub from config
    from_hub = config['data'].get('from_hub', False)

    # Apply command-line overrides if provided
    if num_iterations is not None:
        config['training']['num_iterations'] = num_iterations
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
    if learning_rate is not None:
        config['training']['learning_rate'] = learning_rate

    # Set Modal-specific paths
    config['data']['repo_name'] = "electronickale/cmu-10799-celeba64-subset"
    # Set root path for both modes:
    # - from_hub=true: checks for cached Arrow format first, then downloads from HF
    # - from_hub=false: looks for traditional folder structure (train/images/)
    config['data']['root'] = "/data/celeba"
    config['checkpoint']['dir'] = f"/data/checkpoints/{config_tag}"
    config['logging']['dir'] = f"/data/logs/{config_tag}"

    # For reflow: override pairs_dir to Modal volume path
    if 'reflow' in config:
        pairs_dir = config['reflow'].get('pairs_dir', 'reflow_pairs')
        # Strip leading /data/ or data/ if already present in config
        for prefix in ('/data/', 'data/'):
            if pairs_dir.startswith(prefix):
                pairs_dir = pairs_dir[len(prefix):]
                break
        config['reflow']['pairs_dir'] = f"/data/{pairs_dir}"
        # If init_from_stage1 is set, prepend /data/
        init_path = config['reflow'].get('init_from_stage1')
        if init_path and not init_path.startswith('/data'):
            config['reflow']['init_from_stage1'] = f"/data/{init_path}"

    # Create directories
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)
    os.makedirs(config['logging']['dir'], exist_ok=True)

    resume_path = f"/data/{resume_from}" if resume_from else None

    # Use torchrun for multi-GPU, direct import for single GPU
    if num_gpus > 1:
        temp_config_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                yaml.safe_dump(config, temp_file)
                temp_config_path = temp_file.name

            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                "/root/train.py",
                "--method", method,
                "--config", temp_config_path,
            ]
            if resume_path:
                cmd.extend(["--resume", resume_path])
            if overfit_single_batch:
                cmd.append("--overfit-single-batch")

            subprocess.run(cmd, check=True)
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    else:
        from train import train as run_training
        run_training(method_name=method, config=config, resume_path=resume_path, overfit_single_batch=overfit_single_batch)

    volume.commit()
    return f"Training complete! Checkpoints saved to /data/checkpoints/{method}"


# Create training functions for different GPU counts
@app.function(image=image, gpu="L40S:1", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_1gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:2", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_2gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:3", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_3gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:4", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_4gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:5", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_5gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:6", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_6gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:7", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_7gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

@app.function(image=image, gpu="L40S:8", timeout=60*60*12, volumes={"/data": volume}, secrets=[modal.Secret.from_name("wandb-api-key")])
def train_8gpu(method: str = "ddpm", config_path: str = None, resume_from: str = None, num_iterations: int = None, batch_size: int = None, learning_rate: float = None, overfit_single_batch: bool = False):
    return _train_impl(method, config_path, resume_from, num_iterations, batch_size, learning_rate, overfit_single_batch)

# Map GPU counts to functions
TRAIN_FUNCTIONS = {
    1: train_1gpu,
    2: train_2gpu,
    3: train_3gpu,
    4: train_4gpu,
    5: train_5gpu,
    6: train_6gpu,
    7: train_7gpu,
    8: train_8gpu,
}


# =============================================================================
# Sampling Function
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,  # 3 hours
    volumes={"/data": volume},
)
def sample(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    num_samples: int = None,
    num_steps: int = None,
    seed: int = None,
    sampler: str = None,
    eta: float = None,
):
    """
    Generate samples from a trained model.

    Uses sample.py via subprocess, similar to how training uses train.py.
    """
    import os
    import subprocess
    from datetime import datetime

    # Set up paths
    checkpoint_path = f"/data/{checkpoint}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/data/samples/{method}_{timestamp}.png"

    os.makedirs("/data/samples", exist_ok=True)

    # Build command to run sample.py
    cmd = [
        "python", "/root/sample.py",
        "--checkpoint", checkpoint_path,
        "--method", method,
        "--grid",
        "--output", output_path,
    ]

    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if num_steps is not None:
        cmd.extend(["--num_steps", str(num_steps)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if sampler is not None:
        cmd.extend(["--sampler", str(sampler)])
    if eta is not None:
        cmd.extend(["--eta", str(eta)])

    subprocess.run(cmd, check=True)
    volume.commit()

    return f"Samples saved to {output_path}"


# =============================================================================
# Dataset Download Function
# =============================================================================

@app.function(
    image=image,
    timeout=60 * 60,  # 1 hour
    volumes={"/data": volume},
)
def download_dataset():
    """
    Download the dataset from HuggingFace Hub to Modal volume.

    Caches the dataset in Arrow format at /data/celeba. After downloading,
    training with from_hub=true will automatically use this cached version
    instead of redownloading.
    """
    import sys
    sys.path.insert(0, "/root")

    from datasets import load_dataset
    import os

    print("Downloading dataset from HuggingFace Hub...")
    dataset = load_dataset("electronickale/cmu-10799-celeba64-subset")

    # Save to volume in Arrow format
    os.makedirs("/data/celeba", exist_ok=True)
    dataset.save_to_disk("/data/celeba")

    volume.commit()

    print(f"Dataset cached to /data/celeba")
    print(f"Train set size: {len(dataset['train'])}")
    return "Dataset download complete! Training with from_hub=true with root = '/data/celeba' will now use this cached version."


# =============================================================================
# Evaluation Function (using torch-fidelity)
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,  # 8 hours
    volumes={"/data": volume},
)
def evaluate_torch_fidelity(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    metrics: str = "fid,kid",
    num_samples: int = 5000,
    batch_size: int = 128,
    num_steps: int = None,
    sampler: str = None,
    eta: float = None,
    override: bool = False,
    kp_sample: float = None,
    ki_sample: float = None,
    kd_sample: float = None,
):
    """
    Evaluate using torch-fidelity CLI.

    Uses the fidelity command to compute metrics directly.

    Args:
        method: 'ddpm'
        checkpoint: Path to checkpoint (relative to /data)
        metrics: Comma-separated: 'fid', 'kid', 'is' (default: 'fid,kid')
        num_samples: Number of samples to generate
        batch_size: Batch size
        num_steps: Sampling steps (optional)
        sampler: Sampling algorithm passed to sample.py (optional)
        eta: DDIM eta passed to sample.py (optional)
        override: Force regenerate samples even if they exist
        kp_sample: PID kp for sampling (pid_flow_matching only; None = use training kp)
        ki_sample: PID ki for sampling (pid_flow_matching only; None = use training ki)
        kd_sample: PID kd for sampling (pid_flow_matching only; None = use training kd)
    """
    import sys
    import subprocess
    from pathlib import Path
    sys.path.insert(0, "/root")

    checkpoint_path = f"/data/{checkpoint}"

    # Put samples in same parent dir as checkpoint under samples/
    # Use num_steps-specific subdir so parallel runs at different NFE don't conflict
    checkpoint_dir = Path(checkpoint_path).parent
    steps_tag = f"steps{num_steps}" if num_steps else "steps_default"
    # Include PID sample gains in cache key so different sampling gains don't collide
    if kp_sample is not None or ki_sample is not None or kd_sample is not None:
        gains_tag = f"_kps{kp_sample}_kis{ki_sample}_kds{kd_sample}"
        steps_tag = steps_tag + gains_tag
    generated_dir = str(checkpoint_dir / "samples" / steps_tag / "generated")
    cache_dir = str(checkpoint_dir / "samples" / "cache")

    # Prepare dataset path for torch-fidelity
    # torch-fidelity needs actual image files, not Arrow format
    dataset_arrow_path = "/data/celeba"
    dataset_images_path = "/data/celeba_images"

    # Extract images from Arrow format if not already done
    import os
    if not os.path.exists(dataset_images_path):
        print("=" * 60)
        print("Extracting dataset images for torch-fidelity...")
        print("=" * 60)

        from datasets import load_from_disk

        dataset = load_from_disk(dataset_arrow_path)
        train_data = dataset['train']

        os.makedirs(dataset_images_path, exist_ok=True)

        print(f"Extracting {len(train_data)} images...")
        for idx, item in enumerate(train_data):
            img = item['image']
            img_path = os.path.join(dataset_images_path, f"{idx:06d}.png")
            img.save(img_path)

            if (idx + 1) % 1000 == 0:
                print(f"  Extracted {idx + 1}/{len(train_data)} images")

        volume.commit()
        print(f"Dataset images saved to {dataset_images_path}")
    else:
        print(f"Using cached dataset images at {dataset_images_path}")

    dataset_path = dataset_images_path

    # Step 1: Generate samples
    print("=" * 60)
    print("Step 1/2: Generating samples...")
    print("=" * 60)

    import os
    import shutil
    import glob

    # Check if samples already exist
    need_generation = True
    if os.path.exists(generated_dir) and not override:
        # Check for both png and jpg files
        existing_samples = (
            glob.glob(os.path.join(generated_dir, "*.png")) +
            glob.glob(os.path.join(generated_dir, "*.jpg")) + 
            glob.glob(os.path.join(generated_dir, "*.jpeg"))
        )
        num_existing = len(existing_samples)

        if num_existing >= num_samples:
            print(f"Found {num_existing} existing samples (need {num_samples})")
            print("Skipping sample generation (use --override to force regeneration)")
            need_generation = False
        else:
            print(f"Found {num_existing} existing samples but need {num_samples}")
            print("Regenerating samples...")
            shutil.rmtree(generated_dir)
    elif os.path.exists(generated_dir) and override:
        print("Override flag set, regenerating samples...")
        shutil.rmtree(generated_dir)

    if need_generation:
        sample_cmd = [
            "python", "/root/sample.py",
            "--checkpoint", checkpoint_path,
            "--method", method,
            "--output_dir", generated_dir,
            "--num_samples", str(num_samples),
            "--batch_size", str(batch_size),
        ]

        if num_steps:
            sample_cmd.extend(["--num_steps", str(num_steps)])
        if sampler is not None:
            sample_cmd.extend(["--sampler", str(sampler)])
        if eta is not None:
            sample_cmd.extend(["--eta", str(eta)])
        if kp_sample is not None:
            sample_cmd.extend(["--kp_sample", str(kp_sample)])
        if ki_sample is not None:
            sample_cmd.extend(["--ki_sample", str(ki_sample)])
        if kd_sample is not None:
            sample_cmd.extend(["--kd_sample", str(kd_sample)])

        subprocess.run(sample_cmd, check=True)
        print(f"Generated {num_samples} samples to {generated_dir}")
    else:
        print(f"Using existing samples from {generated_dir}")

    # Step 2: Run fidelity
    print("\n" + "=" * 60)
    print("Step 2/2: Running torch-fidelity...")
    print("=" * 60)

    os.makedirs(cache_dir, exist_ok=True)

    fidelity_cmd = [
        "fidelity",
        "--gpu", "0",
        "--batch-size", str(batch_size),
        "--cache-root", cache_dir,
        "--input1", generated_dir,
        "--input2", dataset_path,
    ]

    if "fid" in metrics:
        fidelity_cmd.append("--fid")
    if "kid" in metrics:
        fidelity_cmd.append("--kid")
    if "is" in metrics or "isc" in metrics:
        fidelity_cmd.append("--isc")

    print(f"\nRunning command: {' '.join(fidelity_cmd)}\n")

    try:
        result = subprocess.run(fidelity_cmd, check=True, capture_output=True, text=True)
        output = result.stdout
        print(output)

        # Parse FID/KID from output and save to JSON on volume for easy retrieval
        import json, re
        from datetime import datetime
        metrics_dict = {
            "checkpoint": checkpoint,
            "num_steps": num_steps,
            "num_samples": num_samples,
            "timestamp": datetime.now().isoformat(),
            "raw_output": output,
        }
        for line in output.splitlines():
            m = re.search(r"frechet_inception_distance:\s*([\d.]+)", line)
            if m:
                metrics_dict["fid"] = float(m.group(1))
            m = re.search(r"kernel_inception_distance_mean:\s*([\d.eE+\-]+)", line)
            if m:
                metrics_dict["kid_mean"] = float(m.group(1))
            m = re.search(r"kernel_inception_distance_std:\s*([\d.eE+\-]+)", line)
            if m:
                metrics_dict["kid_std"] = float(m.group(1))

        # Save per-run JSON next to the generated samples
        result_json_path = os.path.join(
            str(checkpoint_dir / "samples" / steps_tag), "fid_kid_result.json"
        )
        os.makedirs(os.path.dirname(result_json_path), exist_ok=True)
        with open(result_json_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"\nSaved FID/KID result to {result_json_path}")

        volume.commit()
        return output
    except subprocess.CalledProcessError as e:
        # Print the error output to help debug
        print(f"\nError running fidelity command:")
        print(f"Command: {' '.join(fidelity_cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"\nStdout:\n{e.stdout}")
        if e.stderr:
            print(f"\nStderr:\n{e.stderr}")
        raise


# =============================================================================
# Reflow Pair Generation Function
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 6,  # 6 hours
    volumes={"/data": volume},
)
def generate_pairs(
    checkpoint: str = "logs/rectified_flow/checkpoints/rectified_flow_final.pt",
    num_pairs: int = 60000,
    batch_size: int = 256,
    num_steps: int = 50,
    output_dir: str = "reflow_pairs",
    seed: int = 42,
):
    """Generate synthetic (noise, image) pairs for Reflow Stage 2."""
    import subprocess

    checkpoint_path = f"/data/{checkpoint}"
    output_path = f"/data/{output_dir}"

    cmd = [
        "python", "/root/generate_reflow_pairs.py",
        "--checkpoint", checkpoint_path,
        "--num_pairs", str(num_pairs),
        "--batch_size", str(batch_size),
        "--num_steps", str(num_steps),
        "--output_dir", output_path,
        "--seed", str(seed),
    ]

    subprocess.run(cmd, check=True)
    volume.commit()

    return f"Generated {num_pairs} reflow pairs to {output_path}"


# =============================================================================
# Fixed CelebA Eval Pairs (for consistent MSE comparison across ablation models)
# =============================================================================

@app.function(
    image=image,
    gpu=None,
    timeout=60 * 30,  # 30 minutes
    volumes={"/data": volume},
)
def save_celeba_eval_pairs(
    num_pairs: int = 10000,
    output_dir: str = "celeba_eval_pairs",
    seed: int = 42,
    data_dir: str = "/data/celeba",
):
    """
    Generate and save a fixed set of (x0_noise, x1_celeba) pairs for Stage 1 MSE evaluation.

    These pairs are saved once and reused across all ablation models so that
    per-timestep MSE comparisons are on identical data (apples-to-apples).

    Saves:
        {output_dir}/x0_noise.pt    — shape (N, 3, 64, 64), N(0,I), seed=42
        {output_dir}/x1_celeba.pt   — shape (N, 3, 64, 64), first N CelebA train images
        {output_dir}/meta.txt       — num_pairs, seed, date
    """
    import os
    import torch
    from src.data.celeba import create_dataloader

    output_path = f"/data/{output_dir}"
    os.makedirs(output_path, exist_ok=True)

    print(f"Generating {num_pairs} fixed eval pairs (seed={seed}) ...")

    # Fixed noise
    torch.manual_seed(seed)
    x0 = torch.randn(num_pairs, 3, 64, 64)

    # First num_pairs CelebA train images (no shuffle, deterministic order)
    loader = create_dataloader(
        root=data_dir,
        split="train",
        image_size=64,
        batch_size=256,
        num_workers=2,
        pin_memory=False,
        augment=False,
        shuffle=False,
        drop_last=False,
        from_hub=True,
    )
    images = []
    for batch in loader:
        images.append(batch)
        if sum(b.shape[0] for b in images) >= num_pairs:
            break
    x1 = torch.cat(images, dim=0)[:num_pairs]

    torch.save(x0, os.path.join(output_path, "x0_noise.pt"))
    torch.save(x1, os.path.join(output_path, "x1_celeba.pt"))

    from datetime import datetime
    meta = (
        f"num_pairs: {num_pairs}\n"
        f"seed: {seed}\n"
        f"image_size: 64\n"
        f"x0: N(0,I) with torch.manual_seed({seed})\n"
        f"x1: first {num_pairs} CelebA train images (no shuffle)\n"
        f"created: {datetime.now().isoformat()}\n"
    )
    with open(os.path.join(output_path, "meta.txt"), "w") as f:
        f.write(meta)

    volume.commit()
    print(meta)
    return f"Saved {num_pairs} eval pairs to {output_path}"


# =============================================================================
# Trajectory Visualization Function
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 2,  # 2 hours
    volumes={"/data": volume},
)
def visualize(
    checkpoint: str = None,
    checkpoint2: str = None,
    label1: str = "1-Rectified Flow",
    label2: str = "2-Rectified Flow (Reflow)",
    num_trajectories: int = 20,
    num_steps: int = 50,
    output_dir: str = "visualizations",
    seed: int = 42,
):
    """Visualize ODE trajectories: PCA plots, straightness, interpolation grids."""
    import subprocess

    checkpoint_path = f"/data/{checkpoint}"
    output_path = f"/data/{output_dir}"

    cmd = [
        "python", "/root/visualize_trajectories.py",
        "--checkpoint", checkpoint_path,
        "--label1", label1,
        "--label2", label2,
        "--num_trajectories", str(num_trajectories),
        "--num_steps", str(num_steps),
        "--output_dir", output_path,
        "--seed", str(seed),
    ]

    if checkpoint2:
        cmd.extend(["--checkpoint2", f"/data/{checkpoint2}"])

    subprocess.run(cmd, check=True)
    volume.commit()

    return f"Visualizations saved to {output_path}"


# =============================================================================
# Sample Grid at Multiple Step Counts
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,
    volumes={"/data": volume},
)
def sample_grid_multi_steps(
    method: str = "rectified_flow",
    checkpoint: str = None,
    step_counts: str = "1,2,5,10,20,50",
    num_samples: int = 64,
    seed: int = 42,
    output_dir: str = "samples",
):
    """Generate sample grids at multiple step counts for comparison."""
    import os
    import subprocess

    checkpoint_path = f"/data/{checkpoint}"
    output_base = f"/data/{output_dir}"
    os.makedirs(output_base, exist_ok=True)

    steps_list = [int(s.strip()) for s in step_counts.split(",")]

    for nsteps in steps_list:
        output_path = os.path.join(output_base, f"samples_steps{nsteps:03d}.png")
        cmd = [
            "python", "/root/sample.py",
            "--checkpoint", checkpoint_path,
            "--method", method,
            "--grid",
            "--output", output_path,
            "--num_samples", str(num_samples),
            "--num_steps", str(nsteps),
            "--seed", str(seed),
        ]
        print(f"\n{'='*60}")
        print(f"Generating {num_samples} samples at {nsteps} steps...")
        print(f"{'='*60}")
        subprocess.run(cmd, check=True)
        print(f"Saved to {output_path}")

    volume.commit()
    return f"Saved sample grids at steps={step_counts} to {output_base}"


# =============================================================================
# Timestep Error Analysis Function
# =============================================================================

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 2,  # 2 hours
    volumes={"/data": volume},
)
def analyze_timestep_error(
    checkpoint1: str,
    checkpoint2: str = None,
    label1: str = "1-Rectified Flow (Stage 1)",
    label2: str = "2-Rectified Flow (Stage 2)",
    num_bins: int = 20,
    num_batches: int = 300,
    batch_size: int = 128,
    output_dir: str = "analysis/timestep_error",
    eval_pairs_dir: str = None,
):
    """Compute per-timestep velocity MSE for Stage 1 (and optionally Stage 2) RF models."""
    import subprocess

    output_path = f"/data/{output_dir}"
    cmd = [
        "python", "/root/analyze_timestep_error.py",
        "--checkpoint1", f"/data/{checkpoint1}",
        "--label1", label1,
        "--num_bins", str(num_bins),
        "--num_batches", str(num_batches),
        "--batch_size", str(batch_size),
        "--output_dir", output_path,
        "--data_dir", "/data/celeba",
    ]
    if eval_pairs_dir:
        cmd.extend(["--eval_pairs_dir", f"/data/{eval_pairs_dir}"])
    if checkpoint2:
        cmd.extend([
            "--checkpoint2", f"/data/{checkpoint2}",
            "--label2", label2,
            "--reflow_pairs_dir", "/data/reflow_pairs",
        ])

    subprocess.run(cmd, check=True)
    volume.commit()
    return f"Analysis saved to {output_path}"


# =============================================================================
# CLI Entry Points
# =============================================================================

@app.local_entrypoint()
def main(
    action: str = "train",
    method: str = "ddpm",
    config: str = None,
    checkpoint: str = None,
    checkpoint2: str = None,
    iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    num_samples: int = None,
    num_steps: int = None,
    num_pairs: int = None,
    step_counts: str = None,
    seed: int = None,
    sampler: str = None,
    eta: float = None,
    app_id: str = None,
    log_file: str = None,
    loss_csv: str = None,
    loss_png: str = None,
    metrics: str = None,
    output_dir: str = None,
    eval_pairs_dir: str = None,
    num_bins: int = None,
    overfit_single_batch: bool = False,
    override: bool = False,
    kp_sample: float = None,
    ki_sample: float = None,
    kd_sample: float = None,
):
    """
    Main entry point for Modal CLI.

    See docs/QUICKSTART-MODAL.md for usage instructions.

    All parameters are read from config YAML files first, then overridden by command-line arguments.
    """
    # Normalize action name: accept both hyphens and underscores
    action = action.replace("-", "_")

    if action == "download":
        result = download_dataset.remote()
        print(result)
    elif action == "train":
        # Read config to determine GPU count
        import yaml

        local_config_path = config or f"configs/{method}.yaml"
        with open(local_config_path, 'r') as f:
            local_config = yaml.safe_load(f)

        # Get num_gpus from config
        config_device = local_config['infrastructure'].get('device', 'cuda')
        num_gpus = local_config['infrastructure'].get('num_gpus', 1 if config_device == 'cuda' else 0)
        if num_gpus is None:
            num_gpus = 1 if config_device == 'cuda' else 0

        # Get the appropriate training function
        train_fn = TRAIN_FUNCTIONS.get(num_gpus)
        if train_fn is None:
            raise ValueError(
                f"Unsupported num_gpus={num_gpus} in config. "
                f"Supported: 1-8"
            )

        result = train_fn.remote(
            method=method,
            config_path=config,
            resume_from=checkpoint,
            num_iterations=iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            overfit_single_batch=overfit_single_batch,
        )
        print(result)
    elif action == "sample":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"
        result = sample.remote(
            method=method,
            checkpoint=checkpoint,
            num_samples=num_samples,
            num_steps=num_steps,
            seed=seed,
            sampler=sampler,
            eta=eta,
        )
        print(result)
    elif action == "evaluate" or action == "evaluate_torch_fidelity":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"

        eval_kwargs = {
            'method': method,
            'checkpoint': checkpoint,
            'override': override,
        }
        if metrics is not None:
            eval_kwargs['metrics'] = metrics
        if num_samples is not None:
            eval_kwargs['num_samples'] = num_samples
        if batch_size is not None:
            eval_kwargs['batch_size'] = batch_size
        if num_steps is not None:
            eval_kwargs['num_steps'] = num_steps
        if sampler is not None:
            eval_kwargs['sampler'] = sampler
        if eta is not None:
            eval_kwargs['eta'] = eta
        if kp_sample is not None:
            eval_kwargs['kp_sample'] = kp_sample
        if ki_sample is not None:
            eval_kwargs['ki_sample'] = ki_sample
        if kd_sample is not None:
            eval_kwargs['kd_sample'] = kd_sample

        result = evaluate_torch_fidelity.remote(**eval_kwargs)
        print(result)
    elif action == "generate_pairs":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for generate_pairs")
        pair_kwargs = {
            'checkpoint': checkpoint,
        }
        if num_pairs is not None:
            pair_kwargs['num_pairs'] = num_pairs
        if batch_size is not None:
            pair_kwargs['batch_size'] = batch_size
        if num_steps is not None:
            pair_kwargs['num_steps'] = num_steps
        if output_dir is not None:
            pair_kwargs['output_dir'] = output_dir
        if seed is not None:
            pair_kwargs['seed'] = seed
        result = generate_pairs.remote(**pair_kwargs)
        print(result)
    elif action == "visualize":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for visualize")
        viz_kwargs = {
            'checkpoint': checkpoint,
        }
        if checkpoint2 is not None:
            viz_kwargs['checkpoint2'] = checkpoint2
        if output_dir is not None:
            viz_kwargs['output_dir'] = output_dir
        if seed is not None:
            viz_kwargs['seed'] = seed
        if num_steps is not None:
            viz_kwargs['num_steps'] = num_steps
        result = visualize.remote(**viz_kwargs)
        print(result)
    elif action == "sample_multi_steps":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for sample_multi_steps")
        grid_kwargs = {
            'method': method,
            'checkpoint': checkpoint,
        }
        if step_counts is not None:
            grid_kwargs['step_counts'] = step_counts
        if num_samples is not None:
            grid_kwargs['num_samples'] = num_samples
        if seed is not None:
            grid_kwargs['seed'] = seed
        if output_dir is not None:
            grid_kwargs['output_dir'] = output_dir
        result = sample_grid_multi_steps.remote(**grid_kwargs)
        print(result)
    elif action == "plot_loss":
        # Local-only utility: pull Modal app logs and generate a loss curve CSV/PNG
        # Example:
        #   modal run modal_app.py --action plot_loss --app-id ap-XXXX --loss-csv hw_answers/hw1/loss.csv --loss-png hw_answers/hw1/loss.png
        # If Modal app logs don't contain tqdm output (common), you can pass a local log file:
        #   modal run modal_app.py --action plot_loss --log-file /abs/path/to/log.txt ...
        if app_id is None and log_file is None:
            raise ValueError(
                "For --action plot_loss, provide either --app-id (Modal logs) or --log-file (local logs)."
            )

        import csv
        import re
        import subprocess
        from datetime import datetime
        from pathlib import Path

        # Defaults
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if loss_csv is None:
            loss_csv = f"loss_curve_{app_id}_{timestamp}.csv"
        if loss_png is None:
            loss_png = f"loss_curve_{app_id}_{timestamp}.png"

        loss_csv_path = Path(loss_csv)
        loss_png_path = Path(loss_png)
        loss_csv_path.parent.mkdir(parents=True, exist_ok=True)
        loss_png_path.parent.mkdir(parents=True, exist_ok=True)

        # Load logs
        if log_file is not None:
            text = Path(log_file).read_text(errors="ignore")
        else:
            proc = subprocess.run(
                ["modal", "app", "logs", app_id],
                check=True,
                capture_output=True,
                text=True,
            )
            text = proc.stdout

        # Strip ANSI escape codes (colors, cursor moves)
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)

        # Parse lines like:
        #  70%|...| 70099/100000 [...] loss=0.0158, steps/s=4.33
        # Note: tqdm may print multiple progress chunks on a single line; use finditer.
        pat = re.compile(r"(?P<step>\d+)/\d+.*?loss=(?P<loss>[0-9.]+)")
        step_to_loss = {}
        for line in text.splitlines():
            for m in pat.finditer(line):
                step = int(m.group("step"))
                loss = float(m.group("loss"))
                step_to_loss[step] = loss  # keep last occurrence per step

        steps = sorted(step_to_loss.keys())
        if not steps:
            raise RuntimeError(
                "No loss values found in logs. "
                "Expected lines containing '... step/total ... loss=...'. "
                "Try a different --app-id, or use --log-file pointing to a terminal capture that includes tqdm output."
            )

        # Write CSV
        with open(loss_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss"])
            for s in steps:
                w.writerow([s, step_to_loss[s]])

        print(f"Wrote loss CSV to: {loss_csv_path}")
        print(f"Parsed {len(steps)} points from Modal logs for app_id={app_id}")

        # Try to plot if matplotlib exists
        try:
            import matplotlib.pyplot as plt  # type: ignore

            ys = [step_to_loss[s] for s in steps]
            plt.figure(figsize=(6, 4))
            plt.plot(steps, ys, linewidth=1)
            plt.xlabel("step")
            plt.ylabel("train loss")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(loss_png_path, dpi=200)
            plt.close()
            print(f"Wrote loss PNG to: {loss_png_path}")
        except Exception as e:
            print(
                "Could not generate PNG plot (matplotlib missing or failed). "
                f"CSV is still available. Error: {e}"
            )
    elif action == "timestep_analysis":
        if checkpoint is None:
            raise ValueError("--checkpoint is required for timestep_analysis")
        analysis_kwargs = {
            "checkpoint1": checkpoint,
        }
        if checkpoint2 is not None:
            analysis_kwargs["checkpoint2"] = checkpoint2
        if output_dir is not None:
            analysis_kwargs["output_dir"] = output_dir
        if eval_pairs_dir is not None:
            analysis_kwargs["eval_pairs_dir"] = eval_pairs_dir
        if num_bins is not None:
            analysis_kwargs["num_bins"] = num_bins
        result = analyze_timestep_error.remote(**analysis_kwargs)
        print(result)
    elif action == "save_eval_pairs":
        result = save_celeba_eval_pairs.remote(
            num_pairs=num_pairs or 10000,
            output_dir=output_dir or "celeba_eval_pairs",
            seed=seed or 42,
        )
        print(result)
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: download, train, sample, evaluate, generate_pairs, visualize, sample_multi_steps, plot_loss, timestep_analysis, save_eval_pairs")
