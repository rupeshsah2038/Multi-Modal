#!/bin/bash
# Script to run all MedPix configuration experiments sequentially
# Usage: bash tools/run_all_medpix_experiments.sh

set -e  # Exit on error

# Activate conda environment
echo "Activating fedenv environment..."
eval "$(conda shell.bash hook)"
conda activate fedenv

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="/home/rupesh_2421cs03/projects/Federated-KD/Medpix_modular"
CONFIG_DIR="${BASE_DIR}/config"
EXPERIMENT_SCRIPT="${BASE_DIR}/experiments/run.py"

# Get all medpix config files
CONFIGS=(
    "medpix-simple-combined.yaml"
    "medpix-concat_mlp-combined.yaml"
    "medpix-cross_attention-combined.yaml"
    "medpix-gated-combined.yaml"
    "medpix-transformer_concat-combined.yaml"
    "medpix-modality_dropout-combined.yaml"
    "medpix-film-combined.yaml"
    "medpix-energy_aware-combined.yaml"
    "medpix-shomr-combined.yaml"
)

# Track results
SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_CONFIGS=()

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}         MedPix Fusion Module Comparison Experiments${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""
echo "Total experiments to run: ${#CONFIGS[@]}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Run each configuration
for i in "${!CONFIGS[@]}"; do
    CONFIG_FILE="${CONFIGS[$i]}"
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG_FILE}"
    EXPERIMENT_NUM=$((i + 1))
    
    echo -e "${YELLOW}----------------------------------------------------------------------${NC}"
    echo -e "${YELLOW}Experiment ${EXPERIMENT_NUM}/${#CONFIGS[@]}: ${CONFIG_FILE}${NC}"
    echo -e "${YELLOW}----------------------------------------------------------------------${NC}"
    echo "Config path: ${CONFIG_PATH}"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Check if config file exists
    if [ ! -f "${CONFIG_PATH}" ]; then
        echo -e "${RED}✗ Config file not found: ${CONFIG_FILE}${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CONFIGS+=("${CONFIG_FILE} (not found)")
        continue
    fi
    
    # Run the experiment
    if python "${EXPERIMENT_SCRIPT}" "${CONFIG_PATH}"; then
        echo ""
        echo -e "${GREEN}✓ Experiment ${EXPERIMENT_NUM} completed successfully: ${CONFIG_FILE}${NC}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo -e "${RED}✗ Experiment ${EXPERIMENT_NUM} failed: ${CONFIG_FILE}${NC}"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_CONFIGS+=("${CONFIG_FILE}")
        
        # Ask user if they want to continue after failure
        echo ""
        read -p "Continue with remaining experiments? (y/n): " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${YELLOW}Stopping experiments as requested.${NC}"
            break
        fi
    fi
    
    echo ""
    echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Brief pause between experiments
    if [ $EXPERIMENT_NUM -lt ${#CONFIGS[@]} ]; then
        echo "Waiting 3 seconds before next experiment..."
        sleep 3
    fi
done

# Summary
echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}                        EXPERIMENT SUMMARY${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Total experiments: ${#CONFIGS[@]}"
echo -e "${GREEN}Successful: ${SUCCESS_COUNT}${NC}"
echo -e "${RED}Failed: ${FAIL_COUNT}${NC}"
echo ""

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo -e "${RED}Failed experiments:${NC}"
    for failed_config in "${FAILED_CONFIGS[@]}"; do
        echo "  - ${failed_config}"
    done
    echo ""
fi

# List all fusion modules tested
echo "Fusion modules tested:"
echo "  1. simple"
echo "  2. concat_mlp"
echo "  3. cross_attention"
echo "  4. gated"
echo "  5. transformer_concat"
echo "  6. modality_dropout"
echo "  7. film"
echo "  8. energy_aware_adaptive"
echo "  9. shomr"
echo ""

echo "Results are saved in individual log directories:"
echo "  logs/medpix-vit-base-512-{fusion_module}-combined/"
echo ""
echo -e "${BLUE}======================================================================${NC}"

# Exit with appropriate code
if [ ${FAIL_COUNT} -gt 0 ]; then
    exit 1
else
    exit 0
fi
