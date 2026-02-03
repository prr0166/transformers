#!/bin/bash

# Script to run tensor parallel (TP) tests for MoE models
# Tests are run sequentially as each TP test uses 2 GPUs internally
# Usage: ./run_moe_tests.sh /path/to/results

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
DIM='\033[0;90m'
NC='\033[0m' # No Color

# Number of GPUs required for TP tests
NUM_GPUS=2

# Define models to test (model_name -> test_file)
declare -A MODELS=(
    ["afmoe"]="tests/models/afmoe/test_modeling_afmoe.py"
    ["aria"]="tests/models/aria/test_modeling_aria.py"
    ["dbrx"]="tests/models/dbrx/test_modeling_dbrx.py"
    ["deepseek_v2"]="tests/models/deepseek_v2/test_modeling_deepseek_v2.py"
    ["deepseek_v3"]="tests/models/deepseek_v3/test_modeling_deepseek_v3.py"
    ["dots1"]="tests/models/dots1/test_modeling_dots1.py"
    ["ernie4_5_moe"]="tests/models/ernie4_5_moe/test_modeling_ernie4_5_moe.py"
    ["ernie4_5_vl_moe"]="tests/models/ernie4_5_vl_moe/test_modeling_ernie4_5_vl_moe.py"
    ["flex_olmo"]="tests/models/flex_olmo/test_modeling_flex_olmo.py"
    ["glm4_moe"]="tests/models/glm4_moe/test_modeling_glm4_moe.py"
    ["glm4_moe_lite"]="tests/models/glm4_moe_lite/test_modeling_glm4_moe_lite.py"
    ["glm4v_moe"]="tests/models/glm4v_moe/test_modeling_glm4v_moe.py"
    ["gpt_oss"]="tests/models/gpt_oss/test_modeling_gpt_oss.py"
    ["granitemoe"]="tests/models/granitemoe/test_modeling_granitemoe.py"
    ["granitemoehybrid"]="tests/models/granitemoehybrid/test_modeling_granitemoehybrid.py"
    ["granitemoeshared"]="tests/models/granitemoeshared/test_modeling_granitemoeshared.py"
    ["hunyuan_v1_moe"]="tests/models/hunyuan_v1_moe/test_modeling_hunyuan_v1_moe.py"
    ["jamba"]="tests/models/jamba/test_modeling_jamba.py"
    ["jetmoe"]="tests/models/jetmoe/test_modeling_jetmoe.py"
    ["lfm2_moe"]="tests/models/lfm2_moe/test_modeling_lfm2_moe.py"
    ["llama4"]="tests/models/llama4/test_modeling_llama4.py"
    ["longcat_flash"]="tests/models/longcat_flash/test_modeling_longcat_flash.py"
    ["minimax"]="tests/models/minimax/test_modeling_minimax.py"
    ["minimax_m2"]="tests/models/minimax_m2/test_modeling_minimax_m2.py"
    ["mixtral"]="tests/models/mixtral/test_modeling_mixtral.py"
    ["nllb_moe"]="tests/models/nllb_moe/test_modeling_nllb_moe.py"
    ["olmoe"]="tests/models/olmoe/test_modeling_olmoe.py"
    ["phimoe"]="tests/models/phimoe/test_modeling_phimoe.py"
    ["qwen2_moe"]="tests/models/qwen2_moe/test_modeling_qwen2_moe.py"
    ["qwen3_moe"]="tests/models/qwen3_moe/test_modeling_qwen3_moe.py"
    ["qwen3_next"]="tests/models/qwen3_next/test_modeling_qwen3_next.py"
    ["qwen3_omni_moe"]="tests/models/qwen3_omni_moe/test_modeling_qwen3_omni_moe.py"
    ["qwen3_vl_moe"]="tests/models/qwen3_vl_moe/test_modeling_qwen3_vl_moe.py"
    ["solar_open"]="tests/models/solar_open/test_modeling_solar_open.py"
    ["switch_transformers"]="tests/models/switch_transformers/test_modeling_switch_transformers.py"
)

# Check that we have at least 2 GPUs
AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "Need at least $NUM_GPUS GPUs for TP tests, but only $AVAILABLE_GPUS detected!"
    exit 1
fi
echo "Using $NUM_GPUS GPUs for TP tests (available: $AVAILABLE_GPUS)"

# Handle results directory - use provided path or create temp directory
if [ -n "$1" ]; then
    RESULTS_DIR="$1"
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
elif [ -n "$RESULTS_DIR" ]; then
    # RESULTS_DIR already set via environment variable
    mkdir -p "$RESULTS_DIR"
    CLEANUP_RESULTS=false
else
    RESULTS_DIR=$(mktemp -d)
    CLEANUP_RESULTS=true
fi

# Only cleanup if we created a temp directory
if [ "$CLEANUP_RESULTS" = true ]; then
    trap "rm -rf $RESULTS_DIR" EXIT
fi

echo "Results directory: $RESULTS_DIR"

echo "=========================================="
echo "  MoE Models TP Test Script"
echo "  (Sequential execution using $NUM_GPUS GPUs)"
echo "=========================================="
echo ""

# Function to run TP pytest tests
run_test() {
    local model_name=$1
    local test_file=$2
    local result_file="$RESULTS_DIR/${model_name}.result"
    
    echo -e "${YELLOW}Starting: ${model_name} (${test_file})${NC}"
    
    # Run only tensor parallel tests using first 2 GPUs
    CUDA_VISIBLE_DEVICES=0,1 \
        python -m pytest -v "$test_file" -k "test_tensor_parallel" \
        > "$RESULTS_DIR/${model_name}.log" 2>&1
    
    local exit_code=$?
    
    # Write result to file (for collection later)
    if [ $exit_code -eq 0 ]; then
        echo "SUCCESS" > "$result_file"
        echo -e "${GREEN}✓ ${model_name}: SUCCESS${NC}"
    else
        echo "FAILED (exit code: $exit_code)" > "$result_file"
        echo -e "${RED}✗ ${model_name}: FAILED (exit code: $exit_code)${NC}"
    fi
}

# Convert associative array keys to indexed array for scheduling
MODEL_NAMES=(${!MODELS[@]})
NUM_MODELS=${#MODEL_NAMES[@]}

# Run tests sequentially (each TP test uses 2 GPUs internally)
for model_name in "${MODEL_NAMES[@]}"; do
    test_file="${MODELS[$model_name]}"
    run_test "$model_name" "$test_file"
done

# Print summary
echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo ""

success_count=0
fail_count=0

for model_name in "${MODEL_NAMES[@]}"; do
    result_file="$RESULTS_DIR/${model_name}.result"
    if [ -f "$result_file" ]; then
        result=$(cat "$result_file")
        if [[ "$result" == "SUCCESS" ]]; then
            echo -e "${GREEN}✓ ${model_name}: ${result}${NC}"
            ((success_count++))
        else
            echo -e "${RED}✗ ${model_name}: ${result}${NC}"
            # Show last few lines of error
            echo -e "${DIM}  Error snippet:"
            tail -n 5 "$RESULTS_DIR/${model_name}.log" | while read -r line; do echo -e "    ${DIM}${line}${NC}"; done
            ((fail_count++))
        fi
    else
        echo -e "${RED}✗ ${model_name}: NO RESULT (test may have crashed)${NC}"
        ((fail_count++))
    fi
done

echo ""
echo "-------------------------------------------"
echo -e "Total: ${GREEN}${success_count} passed${NC}, ${RED}${fail_count} failed${NC}"
echo "=========================================="

# Show logs for failed tests
if [ $fail_count -gt 0 ]; then
    echo ""
    echo "Failed test logs available in: $RESULTS_DIR"
    echo "To view: cat $RESULTS_DIR/<model_name>.log"
fi

# Exit with failure if any tests failed
if [ $fail_count -gt 0 ]; then
    exit 1
fi