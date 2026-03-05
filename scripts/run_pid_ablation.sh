#!/bin/bash
# =============================================================================
# PID-FM Component & Scale Ablation Pipeline
# =============================================================================
#
# Ablation design: fix K (kp=0.5, ki=0.1, kd=0.05), vary sigma independently.
# kp_sample = kp (training and sampling use same gains).
#
# Effective parameters:
#   P_eff = kp * sigma_p
#   I_eff = ki * sigma_i
#   D_eff = kd * sigma_d
#
# Baseline (DONE): P_eff=0.010, I_eff=0.001, D_eff=0.00025, FID@50=8.1
#   (LN-PID-FM, logs/ln_pid_flow_matching/pid_flow_matching_20260227_044335)
#
# New experiments (6 runs): vary ONE parameter at a time
#   pid_ln_P0     P_eff=0     I_eff=0.001  D_eff=0.00025
#   pid_ln_P5x    P_eff=0.050 I_eff=0.001  D_eff=0.00025
#   pid_ln_I0     P_eff=0.010 I_eff=0     D_eff=0.00025
#   pid_ln_I5x    P_eff=0.010 I_eff=0.005 D_eff=0.00025
#   pid_ln_D0     P_eff=0.010 I_eff=0.001  D_eff=0
#   pid_ln_D5x    P_eff=0.010 I_eff=0.001  D_eff=0.00125
#
# Usage:
#   ./scripts/run_pid_ablation.sh          # train all + eval
#   ./scripts/run_pid_ablation.sh eval     # eval only (set CKPT_* env vars)
#
# Skip training with checkpoint overrides:
#   CKPT_P0="logs/pid_flow_matching/.../pid_flow_matching_final.pt" \
#   CKPT_P5X="..." \
#   ... \
#   ./scripts/run_pid_ablation.sh eval
# =============================================================================

set -e
set -o pipefail

MODE="${1:-all}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/pid_ablation_${TIMESTAMP}"
RESULTS_DIR="hw_answers/hw4/pid_ablation"
SUMMARY_CSV="${RESULTS_DIR}/ablation_results.csv"

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PID Component & Scale Ablation${NC}"
echo -e "${BLUE}  Mode:    ${MODE}${NC}"
echo -e "${BLUE}  Started: $(date)${NC}"
echo -e "${BLUE}========================================${NC}"

# ─────────────────────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────────────────────
EXPERIMENTS=("P0" "P5x" "I0" "I5x" "D0" "D5x")

declare -A CONFIGS=(
    ["P0"]="configs/pid-fm-ln-P0.yaml"
    ["P5x"]="configs/pid-fm-ln-P5x.yaml"
    ["I0"]="configs/pid-fm-ln-I0.yaml"
    ["I5x"]="configs/pid-fm-ln-I5x.yaml"
    ["D0"]="configs/pid-fm-ln-D0.yaml"
    ["D5x"]="configs/pid-fm-ln-D5x.yaml"
)

declare -A DESCRIPTIONS=(
    ["P0"]="P_eff=0    I_eff=0.001 D_eff=0.00025"
    ["P5x"]="P_eff=0.05 I_eff=0.001 D_eff=0.00025"
    ["I0"]="P_eff=0.01 I_eff=0     D_eff=0.00025"
    ["I5x"]="P_eff=0.01 I_eff=0.005 D_eff=0.00025"
    ["D0"]="P_eff=0.01 I_eff=0.001 D_eff=0"
    ["D5x"]="P_eff=0.01 I_eff=0.001 D_eff=0.00125"
)

# Checkpoint overrides (set via env var to skip training)
declare -A CKPT_OVERRIDES=(
    ["P0"]="${CKPT_P0:-}"
    ["P5x"]="${CKPT_P5X:-}"
    ["I0"]="${CKPT_I0:-}"
    ["I5x"]="${CKPT_I5X:-}"
    ["D0"]="${CKPT_D0:-}"
    ["D5x"]="${CKPT_D5X:-}"
)

# ─────────────────────────────────────────────────────────────
# Helper: find latest checkpoint
# ─────────────────────────────────────────────────────────────
find_latest_checkpoint() {
    local base_dir=$1
    local run_dir
    run_dir=$(modal volume ls cmu-10799-diffusion-data "${base_dir}/" 2>/dev/null \
        | grep -v '^$' | sort | tail -1 | awk '{print $1}' | sed 's:/*$::')
    if [ -n "${run_dir}" ]; then
        echo "${run_dir}/checkpoints/pid_flow_matching_final.pt"
    fi
}

# ─────────────────────────────────────────────────────────────
# Step 0: Download dataset
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}  Step 0: Download dataset${NC}"
modal run modal_app.py --action download 2>&1 | tee "${LOG_DIR}/step0_download.log"

# ─────────────────────────────────────────────────────────────
# Step 1: Train all 6 variants in parallel
# ─────────────────────────────────────────────────────────────
if [ "${MODE}" != "eval" ]; then
    echo ""
    echo -e "${YELLOW}  Step 1: Training 6 variants in parallel${NC}"
    for exp in "${EXPERIMENTS[@]}"; do
        echo -e "${BLUE}    ${exp}: ${DESCRIPTIONS[$exp]}${NC}"
    done
    echo ""

    declare -A TRAIN_PIDS

    for exp in "${EXPERIMENTS[@]}"; do
        override="${CKPT_OVERRIDES[$exp]}"
        if [ -n "${override}" ]; then
            echo -e "${CYAN}  ⏭ ${exp}: using checkpoint override${NC}"
            continue
        fi

        config="${CONFIGS[$exp]}"
        train_log="${LOG_DIR}/train_${exp}.log"
        echo -e "${BLUE}  ▶ Launching: ${exp}  (${config})${NC}"

        modal run modal_app.py \
            --action train \
            --method pid_flow_matching \
            --config "${config}" \
            > "${train_log}" 2>&1 &

        TRAIN_PIDS["${exp}"]=$!
    done

    echo -e "${BLUE}  All jobs launched. Waiting...${NC}"

    TRAIN_FAILED=0
    for exp in "${EXPERIMENTS[@]}"; do
        pid_val="${TRAIN_PIDS[$exp]:-}"
        [ -z "${pid_val}" ] && continue
        if wait "${pid_val}"; then
            echo -e "${GREEN}  ✓ ${exp} done${NC}"
        else
            echo -e "${RED}  ✗ ${exp} FAILED — see ${LOG_DIR}/train_${exp}.log${NC}"
            TRAIN_FAILED=1
        fi
    done

    [ "${TRAIN_FAILED}" -eq 1 ] && \
        echo -e "${RED}  WARNING: some training runs failed. Continuing to eval...${NC}"
fi

# ─────────────────────────────────────────────────────────────
# Step 2: Resolve checkpoint paths
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}  Step 2: Resolving checkpoints${NC}"

declare -A CKPTS

for exp in "${EXPERIMENTS[@]}"; do
    override="${CKPT_OVERRIDES[$exp]}"
    if [ -n "${override}" ]; then
        CKPTS["${exp}"]="${override}"
        echo -e "${CYAN}  ${exp}: ${override} (override)${NC}"
    else
        ckpt=$(find_latest_checkpoint "logs/pid_flow_matching")
        if [ -z "${ckpt}" ]; then
            echo -e "${RED}  ERROR: cannot find checkpoint for ${exp}${NC}"
            exit 1
        fi
        CKPTS["${exp}"]="${ckpt}"
        echo -e "${GREEN}  ${exp}: ${ckpt}${NC}"
    fi
done

# Save checkpoint paths
for exp in "${EXPERIMENTS[@]}"; do
    echo "${exp}=${CKPTS[$exp]}"
done > "${LOG_DIR}/checkpoint_paths.txt"

# ─────────────────────────────────────────────────────────────
# Step 3: Evaluate at NFE=1,2,5,10,20,50 (5000 samples)
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}  Step 3: Evaluating — 5000 samples each${NC}"

echo "model,nfe,fid,kid" > "${SUMMARY_CSV}"

for exp in "${EXPERIMENTS[@]}"; do
    ckpt="${CKPTS[$exp]}"
    echo ""
    echo -e "${BLUE}  ── Evaluating: ${exp} ──${NC}"

    for nfe in 1 2 5 10 20 50; do
        eval_log="${LOG_DIR}/eval_${exp}_nfe${nfe}.log"

        modal run modal_app.py \
            --action evaluate \
            --method pid_flow_matching \
            --checkpoint "${ckpt}" \
            --metrics "fid,kid" \
            --num-samples 5000 \
            --num-steps "${nfe}" \
            --override \
            > "${eval_log}" 2>&1

        fid=$(grep -oE 'frechet_inception_distance: [0-9.]+' "${eval_log}" \
            | awk '{print $2}' | tail -1)
        kid=$(grep -oE 'kernel_inception_distance_mean: [0-9.]+' "${eval_log}" \
            | awk '{print $2}' | tail -1)

        echo "${exp},${nfe},${fid:-N/A},${kid:-N/A}" >> "${SUMMARY_CSV}"
        echo -e "${GREEN}    NFE=${nfe}  FID=${fid:-N/A}  KID=${kid:-N/A}${NC}"
    done
done

# ─────────────────────────────────────────────────────────────
# Step 4: Print summary table
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PID Ablation — FID@50 (↓), 5k samples${NC}"
echo -e "${GREEN}  Fixed: LN mu=0,sigma=1 | K fixed${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

printf "  %-12s  %-30s  %7s  %7s  %7s  %7s  %7s  %7s\n" \
    "Exp" "P_eff / I_eff / D_eff" "NFE=1" "NFE=2" "NFE=5" "NFE=10" "NFE=20" "NFE=50"
printf "  %-12s  %-30s  %7s  %7s  %7s  %7s  %7s  %7s\n" \
    "------------" "------------------------------" "-------" "-------" "-------" "-------" "-------" "-------"

# Reference baselines
printf "  %-12s  %-30s  %7s  %7s  %7s  %7s  %7s  %7s\n" \
    "no-pid" "P=0    I=0    D=0" "230.2" "95.6" "34.7" "18.4" "12.8" "9.7"
printf "  %-12s  %-30s  %7s  %7s  %7s  %7s  %7s  %7s\n" \
    "baseline" "P=0.010 I=0.001 D=2.5e-4" "229.4" "90.9" "30.9" "16.0" "11.0" "8.1"
printf "  %-12s  %-30s  %7s  %7s  %7s  %7s  %7s  %7s\n" \
    "------------" "------------------------------" "-------" "-------" "-------" "-------" "-------" "-------"

for exp in "${EXPERIMENTS[@]}"; do
    row=()
    for nfe in 1 2 5 10 20 50; do
        fid=$(grep "^${exp},${nfe}," "${SUMMARY_CSV}" | cut -d, -f3)
        row+=("${fid:-N/A}")
    done
    printf "  %-12s  %-30s  %7s  %7s  %7s  %7s  %7s  %7s\n" \
        "${exp}" "${DESCRIPTIONS[$exp]}" \
        "${row[0]}" "${row[1]}" "${row[2]}" "${row[3]}" "${row[4]}" "${row[5]}"
done

echo ""
echo -e "${GREEN}  CSV: ${SUMMARY_CSV}${NC}"
echo -e "${GREEN}  Logs: ${LOG_DIR}/${NC}"
echo -e "${GREEN}  Done! $(date)${NC}"
echo -e "${GREEN}========================================${NC}"
