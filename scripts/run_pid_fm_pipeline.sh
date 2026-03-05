#!/bin/bash
# =============================================================================
# PID Flow Matching Pipeline (Modal)
# =============================================================================
#
# Only trains and evaluates PID-FM. FM baseline already exists — compare
# results against the existing RF/FM logs manually.
#
# Uses same settings as run_rectified_flow_pipeline.sh for fair comparison:
#   - DiT-S/4, 200K iters
#   - Sample grids at NFE=1,2,5,10,20,50 (64 samples)
#   - Evaluate KID+FID at NFE=1,2,5,10,20,50 (1000 samples)
#
# Pipeline:
#   0. Download dataset (if needed)
#   1. Train PID-FM (DiT-S, 200K iters)
#   2. Generate sample grids at NFE=1,2,5,10,20,50
#   3. Evaluate KID+FID at NFE=1,2,5,10,20,50
#   4. Download results + save summary CSV/TXT
#
# Usage:
#   ./scripts/run_pid_fm_pipeline.sh              # Run all steps (0→4)
#   ./scripts/run_pid_fm_pipeline.sh 2             # Resume from step 2
#
# To reproduce from a known checkpoint (skips training):
#   PID_CKPT_OVERRIDE="logs/pid_flow_matching/pid_flow_matching_20260220_013438/checkpoints/pid_flow_matching_final.pt" \
#   ./scripts/run_pid_fm_pipeline.sh 2
#
# =============================================================================

set -e
set -o pipefail

START_STEP=${1:-0}

# ─────────────────────────────────────────────────────────────
# Checkpoint override — set via env var to skip training
# ─────────────────────────────────────────────────────────────
PID_CKPT_OVERRIDE="${PID_CKPT_OVERRIDE:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/pid_pipeline_${TIMESTAMP}"
RESULTS_DIR="hw_answers/hw4"
SUMMARY_CSV="${RESULTS_DIR}/eval_results.csv"
SUMMARY_TXT="${RESULTS_DIR}/eval_results.txt"

mkdir -p "${LOG_DIR}"
mkdir -p "${RESULTS_DIR}/samples"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PID Flow Matching Pipeline${NC}"
echo -e "${BLUE}  Started: $(date)${NC}"
echo -e "${BLUE}  Logs:    ${LOG_DIR}/${NC}"
echo -e "${BLUE}  Results: ${RESULTS_DIR}/${NC}"
echo -e "${BLUE}  Starting from step: ${START_STEP}${NC}"
[ -n "${PID_CKPT_OVERRIDE}" ] && \
    echo -e "${CYAN}  Checkpoint override: ${PID_CKPT_OVERRIDE}${NC}"
echo -e "${BLUE}========================================${NC}"

# ---------- Helpers ----------

run_step() {
    local step_num=$1
    local step_name=$2
    local log_file="${LOG_DIR}/step${step_num}_${step_name}.log"
    shift 2

    if [ "${step_num}" -lt "${START_STEP}" ] 2>/dev/null; then
        echo -e "${BLUE}  ⏭ Skipping step ${step_num} (${step_name})${NC}"
        return 0
    fi

    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Step ${step_num}: ${step_name}${NC}"
    echo -e "${YELLOW}  $(date)${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if "$@" 2>&1 | tee "${log_file}"; then
        echo -e "${GREEN}  ✓ Step ${step_num} completed — log: ${log_file}${NC}"
        return 0
    else
        echo -e "${RED}  ✗ Step ${step_num} FAILED! Check ${log_file}${NC}"
        return 1
    fi
}

find_latest_checkpoint() {
    local base_dir=$1
    local filename=$2
    local run_dir
    run_dir=$(modal volume ls cmu-10799-diffusion-data "${base_dir}/" 2>/dev/null \
        | grep -v '^$' \
        | sort \
        | tail -1 \
        | sed 's:/*$::')
    if [ -n "${run_dir}" ]; then
        echo "${run_dir}/checkpoints/${filename}"
    fi
}

# Parse FID and KID from a step log file (torch-fidelity output format)
# Outputs: "fid,kid"
parse_metrics() {
    local log_file=$1
    local fid kid
    fid=$(grep -oE 'frechet_inception_distance: [0-9.]+' "${log_file}" \
        | awk '{print $2}' | tail -1)
    kid=$(grep -oE 'kernel_inception_distance_mean: [0-9.]+' "${log_file}" \
        | awk '{print $2}' | tail -1)
    echo "${fid:-N/A},${kid:-N/A}"
}

# ─────────────────────────────────────────────────────────────
# Step 0: Download dataset
# ─────────────────────────────────────────────────────────────
run_step 0 "download_dataset" \
    modal run modal_app.py --action download

# ─────────────────────────────────────────────────────────────
# Step 1: Train PID-FM (DiT-S, 200K iters)
# ─────────────────────────────────────────────────────────────
if [ -n "${PID_CKPT_OVERRIDE}" ]; then
    echo -e "${CYAN}\nSkipping training: using provided checkpoint${NC}"
    PID_CKPT="${PID_CKPT_OVERRIDE}"
else
    run_step 1 "train_pid_fm" \
        modal run modal_app.py \
            --action train \
            --method pid_flow_matching \
            --config configs/pid-fm.yaml

    # Auto-find the latest checkpoint
    echo ""
    echo -e "${BLUE}Finding PID-FM checkpoint...${NC}"
    PID_CKPT=$(find_latest_checkpoint "logs/pid_flow_matching" "pid_flow_matching_final.pt")
    if [ -z "${PID_CKPT}" ]; then
        echo -e "${RED}ERROR: Could not find PID-FM checkpoint${NC}"
        echo "Try: modal volume ls cmu-10799-diffusion-data logs/pid_flow_matching/"
        exit 1
    fi
fi

echo -e "${GREEN}  PID-FM checkpoint: ${PID_CKPT}${NC}"

# Persist checkpoint path for easy reproduction
echo "${PID_CKPT}" > "${LOG_DIR}/checkpoint_path.txt"
echo -e "${BLUE}  Saved to ${LOG_DIR}/checkpoint_path.txt${NC}"

# ─────────────────────────────────────────────────────────────
# Step 2: Generate sample grids at NFE=1,2,5,10,20,50
# ─────────────────────────────────────────────────────────────
run_step 2 "sample_pid_fm" \
    modal run modal_app.py \
        --action sample_multi_steps \
        --method pid_flow_matching \
        --checkpoint "${PID_CKPT}" \
        --step-counts "1,2,5,10,20,50" \
        --num-samples 64 \
        --seed 42 \
        --output-dir "samples/pid_fm"

# ─────────────────────────────────────────────────────────────
# Step 3: Evaluate KID+FID at each NFE — save per-NFE logs
# ─────────────────────────────────────────────────────────────

# Initialize summary CSV
echo "nfe,fid,kid" > "${SUMMARY_CSV}"

for nfe in 1 2 5 10 20 50; do
    echo ""
    echo -e "${YELLOW}=== PID-FM @ NFE=${nfe} ===${NC}"

    eval_log="${LOG_DIR}/step3_eval_pid_fm_nfe${nfe}.log"

    run_step 3 "eval_pid_fm_nfe${nfe}" \
        modal run modal_app.py \
            --action evaluate \
            --method pid_flow_matching \
            --checkpoint "${PID_CKPT}" \
            --metrics "kid,fid" \
            --num-samples 1000 \
            --num-steps "${nfe}" \
            --override

    # Parse FID/KID from this step's log and append to CSV
    metrics=$(parse_metrics "${eval_log}")
    fid_val=$(echo "${metrics}" | cut -d, -f1)
    kid_val=$(echo "${metrics}" | cut -d, -f2)
    echo "${nfe},${metrics}" >> "${SUMMARY_CSV}"
    echo -e "${GREEN}  NFE=${nfe} → FID=${fid_val}  KID=${kid_val}${NC}"
done

# ─────────────────────────────────────────────────────────────
# Step 4: Download results to local
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}  Step 4: Downloading results to local${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

echo "  Downloading sample grids..."
modal volume get cmu-10799-diffusion-data samples/pid_fm/ "${RESULTS_DIR}/samples/" 2>&1 \
    || echo -e "${RED}  Warning: failed to download samples${NC}"

# ─────────────────────────────────────────────────────────────
# Print + save summary table
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PID-FM Evaluation Summary${NC}"
echo -e "${GREEN}========================================${NC}"
printf "\n  %-6s  %-12s  %-12s\n" "NFE" "FID" "KID"
printf "  %-6s  %-12s  %-12s\n" "------" "------------" "------------"
while IFS=',' read -r nfe fid kid; do
    [ "${nfe}" = "nfe" ] && continue  # skip header
    printf "  %-6s  %-12s  %-12s\n" "${nfe}" "${fid}" "${kid}"
done < "${SUMMARY_CSV}"
echo ""

# Human-readable summary file
{
    echo "PID Flow Matching — Evaluation Results"
    echo "======================================"
    echo "Checkpoint : ${PID_CKPT}"
    echo "Date       : $(date)"
    echo ""
    printf "%-6s  %-14s  %-14s\n" "NFE" "FID" "KID (mean)"
    printf "%-6s  %-14s  %-14s\n" "------" "--------------" "--------------"
    while IFS=',' read -r nfe fid kid; do
        [ "${nfe}" = "nfe" ] && continue
        printf "%-6s  %-14s  %-14s\n" "${nfe}" "${fid}" "${kid}"
    done < "${SUMMARY_CSV}"
} > "${SUMMARY_TXT}"

echo -e "${GREEN}  CSV:     ${SUMMARY_CSV}${NC}"
echo -e "${GREEN}  Summary: ${SUMMARY_TXT}${NC}"

# ─────────────────────────────────────────────────────────────
# Done!
# ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  PID-FM Pipeline Complete!${NC}"
echo -e "${GREEN}  Finished: $(date)${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Local results:"
echo "  Logs:        ${LOG_DIR}/"
echo "  Samples:     ${RESULTS_DIR}/samples/pid_fm/"
echo "  Eval CSV:    ${SUMMARY_CSV}"
echo "  Eval TXT:    ${SUMMARY_TXT}"
echo ""
echo "Remote checkpoint (Modal volume):"
echo "  ${PID_CKPT}"
echo ""
echo "To reproduce from this checkpoint (skip training):"
echo "  PID_CKPT_OVERRIDE=\"${PID_CKPT}\" ./scripts/run_pid_fm_pipeline.sh 2"
echo ""
echo "Compare with existing FM/RF results:"
echo "  modal volume ls cmu-10799-diffusion-data logs/rectified_flow/"
echo "  modal volume ls cmu-10799-diffusion-data samples/stage1/"
