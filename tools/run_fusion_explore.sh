#!/bin/bash
# Run all fusion-explore experiments for both MedPix and Wound datasets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || exit 1

# Fusion modules to test
FUSION_MODULES=(
    "simple"
    "concat_mlp"
    "cross_attention"
    "gated"
    "transformer_concat"
    "modality_dropout"
    "film"
    "energy_aware_adaptive"
    "shomr"
)

DATASETS=("medpix" "wound")

echo "========================================================================"
echo "Starting fusion-explore experiments"
echo "========================================================================"
echo "Total experiments: $((${#DATASETS[@]} * ${#FUSION_MODULES[@]}))"
echo "Datasets: ${DATASETS[*]}"
echo "Fusion modules: ${FUSION_MODULES[*]}"
echo ""

TOTAL=0
COMPLETED=0
FAILED=0
FAILED_EXPERIMENTS=()

for dataset in "${DATASETS[@]}"; do
    for fusion in "${FUSION_MODULES[@]}"; do
        ((TOTAL++))
        CONFIG_FILE="config/fusion-explore/${dataset}-${fusion}-combined.yaml"
        
        echo ""
        echo "========================================================================"
        echo "Running: ${dataset^^} - ${fusion}"
        echo "Config: ${CONFIG_FILE}"
        echo "========================================================================"
        echo ""
        
        if python experiments/run.py "$CONFIG_FILE"; then
            ((COMPLETED++))
            echo ""
            echo "✓ Completed: ${dataset}-${fusion}"
        else
            ((FAILED++))
            FAILED_EXPERIMENTS+=("${dataset}-${fusion}")
            echo ""
            echo "✗ Failed: ${dataset}-${fusion}"
        fi
    done
done

# Summary
echo ""
echo "========================================================================"
echo "FUSION-EXPLORE EXPERIMENTS SUMMARY"
echo "========================================================================"
echo "Total experiments: $TOTAL"
echo "Completed: $COMPLETED"
echo "Failed: $FAILED"

if [ ${#FAILED_EXPERIMENTS[@]} -gt 0 ]; then
    echo ""
    echo "Failed experiments:"
    for exp in "${FAILED_EXPERIMENTS[@]}"; do
        echo "  - $exp"
    done
fi

echo ""
echo "Results saved in: logs/fusion-explore/"

exit $FAILED
