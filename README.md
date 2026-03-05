# PID Flow Matching

Applying PID (Proportional-Integral-Derivative) control theory to Flow Matching training and sampling for unconditional image generation.

**Task:** Unconditional CelebA 64×64 generation
**Model:** DiT-S/4 (~33M parameters)
**Platform:** Modal (A100 GPU, ~8–9 GPU-hours per training run)

---

## Table of Contents

1. [Running](#1-running)
2. [Method Overview](#2-method-overview)
3. [Experimental Results](#3-experimental-results)

---

## 1. Running

### Project Structure

```
PID_Flow_Matching/
├── train.py                      # Main training loop
├── sample.py                     # Sampling script
├── download_dataset.py           # CelebA dataset download
├── modal_app.py                  # Modal cloud GPU entrypoint
├── configs/                      # YAML configs (one per experiment)
├── scripts/
│   ├── run_pid_fm_pipeline.sh    # End-to-end: train + eval
│   ├── run_pid_ablation.sh       # P/I/D ablation runs (parallel)
│   └── eval_all_nfe.sh           # Evaluate at NFE=1/2/5/10/20/50
├── src/
│   ├── methods/
│   │   ├── pid_flow_matching.py  # Core PID-FM implementation
│   │   ├── rectified_flow.py     # Flow Matching baseline
│   │   └── flow_matching.py      # Standard Flow Matching
│   ├── models/dit.py             # DiT-S/4 architecture
│   ├── data/celeba.py            # CelebA 64x64 dataloader
│   └── utils/ema.py              # Exponential Moving Average
└── environments/requirements.txt
```

### Config Reference

| File | Model | t-schedule | PID |
|------|-------|-----------|-----|
| `configs/fm.yaml` | FM | uniform | none |
| `configs/fm-ln.yaml` | FM-LN | logit-normal (μ=0, σ=1) | none |
| `configs/pid-fm.yaml` | PID-FM | uniform | kp=0.5, ki=0.1, kd=0.05 |
| `configs/pid-fm-ln.yaml` | **PID-FM-LN** | logit-normal (μ=0, σ=1) | kp=0.5, ki=0.1, kd=0.05 |
| `configs/pid-fm-ln-P0.yaml` | ablation P=0 | logit-normal | kp=0 |
| `configs/pid-fm-ln-P5x.yaml` | ablation P×5 | logit-normal | kp=2.5 |
| `configs/pid-fm-ln-I0.yaml` | ablation I=0 | logit-normal | ki=0 |
| `configs/pid-fm-ln-I5x.yaml` | ablation I×5 | logit-normal | ki=0.5 |
| `configs/pid-fm-ln-D0.yaml` | ablation D=0 | logit-normal | kd=0 |
| `configs/pid-fm-ln-D5x.yaml` | ablation D×5 | logit-normal | kd=0.25 |

---

### Option A: Modal (cloud GPU — recommended for full training)

Each 200K-iteration training run takes ~8–9 GPU-hours on A100. Modal provides on-demand GPUs with no setup beyond authentication.

**Step 1 — Install Modal**

```bash
pip install modal
modal setup        # opens browser to authenticate
```

**Step 2 — Create the persistent volume** (first time only)

```bash
modal volume create cmu-10799-diffusion-data
```

This volume persists checkpoints and datasets across all runs.

**Step 3 — Download the dataset**

```bash
modal run modal_app.py --action download
```

**Step 4 — Train**

```bash
# FM baseline (uniform t-sampling)
modal run modal_app.py \
  --action train \
  --method rectified_flow \
  --config configs/fm.yaml

# FM-LN (logit-normal t, μ=0, σ=1)
modal run modal_app.py \
  --action train \
  --method rectified_flow \
  --config configs/fm-ln.yaml

# PID-FM (PID perturbations, uniform t)
modal run modal_app.py \
  --action train \
  --method pid_flow_matching \
  --config configs/pid-fm.yaml

# PID-FM-LN (best model: PID + logit-normal t)
modal run modal_app.py \
  --action train \
  --method pid_flow_matching \
  --config configs/pid-fm-ln.yaml
```

Run in background (detached) for long jobs:

```bash
modal run --detach modal_app.py \
  --action train \
  --method pid_flow_matching \
  --config configs/pid-fm-ln.yaml &
```

Run all 6 PID ablation variants in parallel:

```bash
./scripts/run_pid_ablation.sh
```

**Step 5 — Evaluate**

```bash
# Evaluate a checkpoint at NFE=1,2,5,10,20,50
for nfe in 1 2 5 10 20 50; do
  modal run modal_app.py \
    --action evaluate \
    --method pid_flow_matching \
    --checkpoint "logs/ln_pid_flow_matching/<run_id>/checkpoints/pid_flow_matching_final.pt" \
    --metrics "kid,fid" \
    --num-samples 5000 \
    --num-steps $nfe \
    --override
done

# Or run the full pipeline (train → eval) in one command
./scripts/run_pid_fm_pipeline.sh

# Resume from existing checkpoint (skip training)
PID_CKPT_OVERRIDE="logs/ln_pid_flow_matching/<run_id>/checkpoints/pid_flow_matching_final.pt" \
./scripts/run_pid_fm_pipeline.sh 2

# List checkpoints on the volume
modal volume ls cmu-10799-diffusion-data logs/ln_pid_flow_matching/
```

---

### Option B: Local (CPU/GPU)

Suitable for debugging or if you have a local GPU. Full training (~200K iters) on CPU is not practical.

**Step 1 — Clone the repo**

```bash
git clone https://github.com/<your-username>/PID_Flow_Matching.git
cd PID_Flow_Matching
```

**Step 2 — Create a virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

Verify Python version (3.10+ required):

```bash
python --version
```

**Step 3 — Install dependencies**

GPU with CUDA 12.6 (NVIDIA drivers 560+, H100/A100):

```bash
pip install -r environments/requirements.txt
```

CPU only or Mac (for debugging):

```bash
pip install torch torchvision torchaudio
pip install numpy pillow PyYAML einops tqdm scipy wandb datasets huggingface-hub torch-fidelity
```

> For other CUDA versions (11.8, 12.1, etc.) edit the `--index-url` line in
> `environments/requirements.txt`. See https://pytorch.org/get-started/locally/

**Step 4 — Install the project package**

```bash
pip install -e .
```

**Step 5 — Verify**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**Step 6 — Download dataset and train**

```bash
# Download dataset locally
python download_dataset.py

# Train (local GPU)
python train.py \
  --method pid_flow_matching \
  --config configs/pid-fm-ln.yaml

# Evaluate
python sample.py \
  --method pid_flow_matching \
  --checkpoint checkpoints/pid_flow_matching_final.pt \
  --num-steps 50
```

---

## 2. Method Overview

### Standard Flow Matching (FM)

Flow Matching trains a velocity network $v_\theta$ to predict the straight-line velocity between noise $x_0 \sim \mathcal{N}(0, I)$ and data $x_1$ along a linear interpolation:

$$x_t = (1-t)\, x_0 + t\, x_1, \quad t \sim \mathcal{U}[0,1]$$

$$\mathcal{L}_\text{FM}(\theta) = \mathbb{E}_{t,\, x_0,\, x_1}\!\left[\, \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \,\right]$$

At inference, Euler integration from noise to image:

$$x_{t+\Delta t} = x_t + v_\theta(x_t, t) \cdot \Delta t$$

**Two limitations motivate this work:**

1. **Training waste.** The per-timestep MSE is U-shaped — high at $t \approx 0$ (high-variance noise) and $t \approx 1$ (high-variance data), low in the informative middle. Uniform $t$-sampling wastes compute on unlearnable endpoints.

2. **Inference fragility.** FM trains only at the ideal interpolant $x_t^* = (1-t)x_0 + tx_1$. At inference, finite Euler steps accumulate errors so the actual $x_t$ drifts from $x_t^*$ — but the model has never seen off-trajectory states and cannot correct them.

---

### Logit-Normal t-Sampling (FM-LN)

Replace $t \sim \mathcal{U}[0,1]$ with a logit-normal distribution (following SD3, Esser et al. 2024):

$$\varepsilon \sim \mathcal{N}(\mu,\, \sigma^2), \qquad t = \text{sigmoid}(\varepsilon) = \frac{1}{1 + e^{-\varepsilon}}$$

This concentrates training mass on mid-timesteps where the MSE is smallest and most learnable, reducing wasted compute at the high-variance endpoints. Best setting from a 5-configuration ablation: **$\mu = 0$, $\sigma = 1$** (symmetric around $t = 0.5$).

**Result:** FID $22.5 \to 9.7$ at NFE=50 (−57%).

---

### PID Flow Matching (PID-FM)

The core contribution. Addresses inference fragility by injecting structured perturbations during training — teaching the model to handle realistic ODE integration errors before it encounters them at test time.

#### Training

Given a training pair $(x_0, x_1)$ and sampled $t$, compute the ideal interpolant $x_t^* = (1-t)x_0 + tx_1$, then generate three structured perturbations:

$$\delta_P = \mathcal{N}(0, I) \cdot \sigma_P \cdot \sqrt{t(1-t) + \epsilon}$$

$$\delta_I = \text{smooth}\left(\mathcal{N}(0, I)\right) \cdot \sigma_I \cdot t$$


$$\delta_D = \mathcal{N}(0, I) \cdot \sigma_D$$

where $\text{smooth}(\cdot)$ is a $4\times$ spatial downsample + bilinear upsample, simulating low-frequency spatially-correlated drift.

The three terms cover complementary classes of ODE error:

| Term | Error type | Why this scaling |
|------|-----------|-----------------|
| **P** — proportional | Instantaneous positional error (most common in Euler steps) | $\sqrt{t(1-t)}$ vanishes at endpoints; corrections are hardest to learn near $t=0,1$ |
| **I** — integral | Slowly accumulated drift over many steps | Spatially smooth; scaled by $t$ to grow with trajectory length |
| **D** — derivative | Sudden large error from coarse step sizes | Flat — equally likely at any $t$ |

Feed the perturbed state to the model and train it to output a corrected velocity that compensates for the perturbation:

$$\tilde{x}_t = x_t^* + \delta_P + \delta_I + \delta_D$$

$$v_\text{target} = (x_1 - x_0) - \bigl(k_P\,\delta_P + k_I\,\delta_I + k_D\,\delta_D\bigr)$$

$$\mathcal{L}_\text{PID}(\theta) = \mathbb{E}_{t,\, x_0,\, x_1}\!\left[\, \| v_\theta(\tilde{x}_t,\, t) - v_\text{target} \|^2 \,\right]$$

#### Sampling

At each Euler step, estimate trajectory error and apply closed-loop PID correction:

$$\hat{x}_1 = x_t + (1-t)\cdot v_\theta(x_t, t) \qquad \text{(one-step endpoint estimate)}$$

$$x_\text{ref} = (1-t)\,x_0^\text{init} + t\,\hat{x}_1 \qquad \text{(estimated ideal position)}$$

$$e_P = x_t - x_\text{ref}, \qquad e_I \mathrel{+}= e_P \cdot \Delta t, \qquad e_D = \frac{e_P - e_P^\text{prev}}{\Delta t}$$

$$v_\text{corr} = v_\theta(x_t, t) - \bigl(k_P\,e_P + k_I\,e_I + k_D\,e_D\bigr)$$

$$x_{t+\Delta t} = x_t + v_\text{corr} \cdot \Delta t$$

#### Hyperparameters

Only the effective strength $k \cdot \sigma$ matters for training — $k$ and $\sigma$ are not independently identifiable. The same gains are used at both training and sampling time (unifying them improves FID 8.7 → 8.1 and removes a free parameter).

| Parameter | Value | Effective strength |
|-----------|-------|--------------------|
| $k_P = 0.5$, $\sigma_P = 0.02$ | | $P_\text{eff} = 0.010$ |
| $k_I = 0.1$, $\sigma_I = 0.01$ | | $I_\text{eff} = 0.001$ |
| $k_D = 0.05$, $\sigma_D = 0.005$ | | $D_\text{eff} = 2.5 \times 10^{-4}$ |
| $t$-schedule: logit-normal | $\mu=0$, $\sigma=1$ | |

The $P : I : D$ effective strength ratio $40 : 4 : 1$ reflects the typical hierarchy of ODE integration errors: proportional errors are largest, derivative errors are smallest.

**Result:** FID $22.5 \to 9.3$ at NFE=50 with uniform $t$ (−59%).


---

## 3. Experimental Results

All models: DiT-S/4 (~33M params), 200K iterations, CelebA 64×64 (~30K images).
Evaluation: FID and KID over 5,000 generated samples vs. full CelebA-64 reference (`torch-fidelity`), seed 42.

### Main Results

**FID ↓**

| NFE | FM | FM-LN | PID-FM | **PID-FM-LN** |
|-----|----|-------|--------|--------------|
| 1   | 237.1 | 230.2 | 229.3 | 229.4   |
| 2   | 121.3 | 95.6  | 110.0 | 90.9    |
| 5   | 50.8  | 34.7  | 38.3  | 30.9    |
| 10  | 33.0  | 18.4  | 19.9  | 16.0    |
| 20  | 26.0  | 12.8  | 13.2  | **11.0**|
| 50  | 22.5  | 9.7   | 9.3   | **8.1** |

**KID ↓ (×10⁻³)**

| NFE | FM | FM-LN | PID-FM | **PID-FM-LN** |
|-----|----|-------|--------|--------------|
| 1   | 250.8 | 264.5 | 240.0 | 259.9   |
| 2   | 108.0 | 88.7  | 100.9 | 83.2    |
| 5   | 37.0  | 32.1  | 33.6  | 28.1    |
| 10  | 19.6  | 16.2  | 17.3  | 13.6    |
| 20  | 11.6  | 10.1  | 10.2  | **7.9** |
| 50  | 6.7   | 5.6   | 5.2   | **3.7** |

**PID-FM-LN** achieves FID=8.1, KID=3.7×10⁻³ at NFE=50 — **−64% FID vs. FM baseline**.

---

### Component Ablation: LN × PID

**FID ↓**

| LN Sampling | PID Perturbations | NFE=1 | NFE=5 | NFE=20 | NFE=50 |
|-------------|-------------------|-------|-------|--------|--------|
| No (uniform) | No  | 237.1 | 50.8 | 26.0 | 22.5 |
| No (uniform) | Yes | 229.3 | 38.3 | 13.2 | 9.3  |
| Yes (μ=0, σ=1) | No  | 230.2 | 34.7 | 12.8 | 9.7  |
| Yes (μ=0, σ=1) | Yes | 229.4 | **30.9** | **11.0** | **8.1** |

**KID ↓ (×10⁻³)**

| LN Sampling | PID Perturbations | NFE=1 | NFE=5 | NFE=20 | NFE=50 |
|-------------|-------------------|-------|-------|--------|--------|
| No (uniform) | No  | 250.8 | 37.0 | 11.6 | 6.7 |
| No (uniform) | Yes | 240.0 | 33.6 | 10.2 | 5.2 |
| Yes (μ=0, σ=1) | No  | 264.5 | 32.1 | 10.1 | 5.6 |
| Yes (μ=0, σ=1) | Yes | 259.9 | **28.1** | **7.9** | **3.7** |

LN and PID address orthogonal failure modes — gains are approximately additive.

---

### Logit-Normal (μ, σ) Ablation (no PID)

**FID ↓**

| μ | σ | NFE=1 | NFE=2 | NFE=5 | NFE=10 | NFE=20 | NFE=50 |
|---|---|-------|-------|-------|--------|--------|--------|
| Uniform | — | 237.1 | 121.3 | 50.8 | 33.0 | 26.0 | 22.5 |
| −0.8 | 1.0 | 232.9 | 108.5 | 41.3 | 22.7 | 16.2 | 14.4 |
| −0.4 | 0.5 | **184.8** | **80.7** | **31.0** | 24.1 | 49.6 † | 53.2 † |
| −0.4 | 1.0 | 240.2 | 104.2 | 38.2 | 20.6 | 13.9 | 10.4 |
| −0.4 | 2.0 | 238.8 | 116.0 | 42.8 | 22.9 | 15.3 | 11.2 |
| **0.0** | **1.0** | 230.2 | 95.6 | 34.7 | **18.4** | **12.8** | **9.7** |

**KID ↓ (×10⁻³)**

| μ | σ | NFE=1 | NFE=2 | NFE=5 | NFE=10 | NFE=20 | NFE=50 |
|---|---|-------|-------|-------|--------|--------|--------|
| Uniform | — | 250.8 | 108.0 | 37.0 | 19.6 | 11.6 | 6.7 |
| −0.8 | 1.0 | 255.0 | 100.0 | 37.9 | 19.8 | 12.6 | 9.2 |
| −0.4 | 0.5 | **197.6** | **77.9** | **28.3** | 19.9 | 54.8 † | 59.6 † |
| −0.4 | 1.0 | 270.7 | 96.9 | 34.7 | 18.3 | 10.9 | 6.0 |
| −0.4 | 2.0 | 251.7 | 106.6 | 38.3 | 20.5 | 12.5 | 7.0 |
| **0.0** | **1.0** | 264.5 | 88.7 | 32.1 | **16.2** | **10.1** | **5.6** |

† **Catastrophic:** $\sigma=0.5$ concentrates training on $t \in [0.35, 0.65]$, severely undertrained endpoints. At NFE=20/50, the Euler integrator passes through those timesteps many times — trajectories collapse. Narrower is NOT always better.

---

### PID Sensitivity Ablation (FID ↓, all with logit-normal μ=0, σ=1)

| Config | Description | NFE=1 | NFE=2 | NFE=5 | NFE=10 | NFE=20 | NFE=50 |
|--------|-------------|-------|-------|-------|--------|--------|--------|
| FM-LN (no PID) | baseline | 230.2 | 95.6 | 34.7 | 18.4 | 12.8 | 9.7 |
| **PID-FM-LN** | P+I+D | 229.4 | 90.9 | 30.9 | 16.0 | 11.0 | **8.1** |
| P=0 | $P_\text{eff}=0$ | 226.5 | 98.1 | 33.7 | 17.8 | 12.3 | 9.0 |
| P×5 | $P_\text{eff}=0.050$ | 239.7 | 89.7 | 32.4 | 16.8 | 12.5 | 13.4 |
| I=0 | $I_\text{eff}=0$ | 227.6 | 92.2 | 30.2 | 15.8 | 10.8 | **8.1** |
| I×5 | $I_\text{eff}=0.005$ | **225.2** | **89.6** | **29.6** | 15.9 | **10.7** | **8.1** |
| D=0 | $D_\text{eff}=0$ | 236.7 | 91.8 | 30.9 | 16.7 | 11.4 | 8.5 |
| D×5 | $D_\text{eff}=0.00125$ | 232.2 | 92.1 | 30.3 | **14.3** | 12.5 | 17.3 † |

† **Catastrophic at NFE=50:** D×5 trains on large high-frequency perturbations; at fine step sizes the model still applies large D-corrections → overshooting. Same root cause as $\sigma=0.5$: train-test mismatch in error magnitude.

Key findings:
- **P** is most critical: P=0 degrades NFE=50 FID from 8.1 → 9.0
- **I×5** is the best single config at low-to-medium NFE (NFE=1: 225.2, NFE=5: 29.6)
- **D×5** is best at NFE=10 (14.3) but catastrophic at NFE=50 (17.3)

