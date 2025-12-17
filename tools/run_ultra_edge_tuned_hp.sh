#!/bin/bash

# Run all ultra-edge-tuned-hp experiments on cuda:0
# Usage: bash tools/run_ultra_edge_tuned_hp.sh

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fedenv

# Set GPU
export CUDA_VISIBLE_DEVICES=0

echo "========================================"
echo "Running Ultra-Edge Tuned-HP Experiments"
echo "GPU: cuda:0"
echo "========================================"
echo ""

# MedPix experiments
echo "=== Starting MedPix Experiments ==="
echo ""

echo "[1/8] Running medpix-deit_small-distilbert..."
python experiments/run.py config/ultra-edge-tuned-hp/medpix-mobilevit_small-distilbert.yaml
echo "✓ Completed medpix-deit_small-distilbert"
echo ""

echo "[2/8] Running medpix-deit_small-minilm..."
python experiments/run.py config/ultra-edge-tuned-hp/medpix-mobilevit_small-minilm.yaml
echo "✓ Completed medpix-deit_small-minilm"
echo ""

echo "[3/8] Running medpix-deit_tiny-distilbert..."
python experiments/run.py config/ultra-edge-tuned-hp/medpix-mobilevit_xxs-distilbert.yaml
echo "✓ Completed medpix-deit_tiny-distilbert"
echo ""

echo "[4/8] Running medpix-deit_tiny-minilm..."
python experiments/run.py config/ultra-edge-tuned-hp/medpix-mobilevit_xxs-minilm.yaml
echo "✓ Completed medpix-deit_tiny-minilm"
echo ""

# Wound experiments
echo "=== Starting Wound Experiments ==="
echo ""

echo "[5/8] Running wound-deit_small-distilbert..."
python experiments/run.py config/ultra-edge-tuned-hp/wound-mobilevit_small-distilbert.yaml
echo "✓ Completed wound-deit_small-distilbert"
echo ""

echo "[6/8] Running wound-deit_small-minilm..."
python experiments/run.py config/ultra-edge-tuned-hp/wound-mobilevit_small-minilm.yaml
echo "✓ Completed wound-deit_small-minilm"
echo ""

echo "[7/8] Running wound-deit_tiny-distilbert..."
python experiments/run.py config/ultra-edge-tuned-hp/wound-mobilevit_xxs-distilbert.yaml
echo "✓ Completed wound-deit_tiny-distilbert"
echo ""

echo "[8/8] Running wound-deit_tiny-minilm..."
python experiments/run.py config/ultra-edge-tuned-hp/wound-mobilevit_xxs-minilm.yaml
echo "✓ Completed wound-deit_tiny-minilm"
echo ""

echo "========================================"
echo "All Ultra-Edge Tuned-HP Experiments Completed!"
echo "Results saved in: logs/ultra-edge-tuned-hp/"
echo "========================================"
