#!/bin/bash

# Run all loss-explore experiments with cross-attention fusion
# Usage: ./tools/run_loss_explore.sh

set -e

# Activate conda environment
conda activate fedenv

# Base directory
cd "$(dirname "$0")/.."

echo "========================================"
echo "Loss Exploration Experiment Runner"
echo "Cross-attention fusion + all losses"
echo "========================================"
echo ""

# Wound dataset experiments
WOUND_CONFIGS=(
    "config/loss-explore/wound-cross_attention-vanilla.yaml"
    "config/loss-explore/wound-cross_attention-combined.yaml"
    "config/loss-explore/wound-cross_attention-crd.yaml"
    "config/loss-explore/wound-cross_attention-rkd.yaml"
    "config/loss-explore/wound-cross_attention-mmd.yaml"
)

# MedPix dataset experiments
MEDPIX_CONFIGS=(
    "config/loss-explore/medpix-cross_attention-vanilla.yaml"
    "config/loss-explore/medpix-cross_attention-combined.yaml"
    "config/loss-explore/medpix-cross_attention-crd.yaml"
    "config/loss-explore/medpix-cross_attention-rkd.yaml"
    "config/loss-explore/medpix-cross_attention-mmd.yaml"
)

# Combine all configs
ALL_CONFIGS=("${WOUND_CONFIGS[@]}" "${MEDPIX_CONFIGS[@]}")

TOTAL=${#ALL_CONFIGS[@]}
SUCCESS=0
FAILED=0

echo "Total experiments: $TOTAL"
echo ""

# Run each experiment
for i in "${!ALL_CONFIGS[@]}"; do
    CONFIG="${ALL_CONFIGS[$i]}"
    NUM=$((i + 1))
    
    echo "========================================"
    echo "[$NUM/$TOTAL] Running: $CONFIG"
    echo "========================================"
    
    if python experiments/run.py "$CONFIG"; then
        echo "✓ Success: $CONFIG"
        ((SUCCESS++))
    else
        echo "✗ Failed: $CONFIG"
        ((FAILED++))
        
        # Ask user whether to continue
        read -p "Continue with remaining experiments? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopping execution."
            break
        fi
    fi
    
    echo ""
done

# Print summary
echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Successful: $SUCCESS/$TOTAL"
echo "Failed: $FAILED/$TOTAL"
echo "========================================"
