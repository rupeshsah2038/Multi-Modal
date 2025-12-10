#!/bin/bash

# Run all ultra-edge experiments with mobile student models
# Usage: ./tools/run_ultra_edge.sh

set -e

# Activate conda environment
conda activate fedenv

# Base directory
cd "$(dirname "$0")/.."

echo "========================================"
echo "Ultra-Edge Experiment Runner"
echo "Teacher: ViT-Base + Bio-ClinicalBERT"
echo "Students: MobileViT + Lightweight Text"
echo "========================================"
echo ""

# Wound dataset experiments
WOUND_CONFIGS=(
    "config/ultra-edge/wound-mobilevit_small-distilbert.yaml"
    "config/ultra-edge/wound-mobilevit_small-minilm.yaml"
    "config/ultra-edge/wound-mobilevit_xxs-distilbert.yaml"
    "config/ultra-edge/wound-mobilevit_xxs-minilm.yaml"
)

# MedPix dataset experiments
MEDPIX_CONFIGS=(
    "config/ultra-edge/medpix-mobilevit_small-distilbert.yaml"
    "config/ultra-edge/medpix-mobilevit_small-minilm.yaml"
    "config/ultra-edge/medpix-mobilevit_xxs-distilbert.yaml"
    "config/ultra-edge/medpix-mobilevit_xxs-minilm.yaml"
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
