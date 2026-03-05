#!/bin/bash
set -e
set -o pipefail

STAGE1_CKPT="logs/rectified_flow/rectified_flow_20260214_053610/checkpoints/rectified_flow_final.pt"
STAGE2_CKPT="logs/rectified_flow_reflow/reflow_20260214_191036/checkpoints/reflow_final.pt"

NFES="1 2 5 10 20 50"

echo "============================================"
echo "  FID/KID Evaluation at Multiple NFE Values"
echo "============================================"

for nfe in $NFES; do
    echo ""
    echo "=== Stage 1 (1-RF) @ NFE=${nfe} ==="
    modal run modal_app.py \
        --action evaluate \
        --method rectified_flow \
        --checkpoint "${STAGE1_CKPT}" \
        --metrics "kid,fid" \
        --num-samples 1000 \
        --num-steps "${nfe}" \
        --override

    echo ""
    echo "=== Stage 2 (Reflow) @ NFE=${nfe} ==="
    modal run modal_app.py \
        --action evaluate \
        --method rectified_flow \
        --checkpoint "${STAGE2_CKPT}" \
        --metrics "kid,fid" \
        --num-samples 1000 \
        --num-steps "${nfe}" \
        --override
done

echo ""
echo "============================================"
echo "  All evaluations complete!"
echo "============================================"
