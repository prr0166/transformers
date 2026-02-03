#!/bin/bash

# Script to run tensor parallel (TP) tests for Dense models
# Tests are run sequentially as each TP test uses 2 GPUs internally
# Usage: ./run_dense_tests.sh /path/to/results

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
    ["apertus"]="tests/models/apertus/test_modeling_apertus.py"
    ["arcee"]="tests/models/arcee/test_modeling_arcee.py"
    ["bart"]="tests/models/bart/test_modeling_bart.py"
    ["bigbird_pegasus"]="tests/models/bigbird_pegasus/test_modeling_bigbird_pegasus.py"
    ["bitnet"]="tests/models/bitnet/test_modeling_bitnet.py"
    ["blenderbot"]="tests/models/blenderbot/test_modeling_blenderbot.py"
    ["blenderbot_small"]="tests/models/blenderbot_small/test_modeling_blenderbot_small.py"
    ["bloom"]="tests/models/bloom/test_modeling_bloom.py"
    ["blt"]="tests/models/blt/test_modeling_blt.py"
    ["codegen"]="tests/models/codegen/test_modeling_codegen.py"
    ["cohere"]="tests/models/cohere/test_modeling_cohere.py"
    ["cohere2"]="tests/models/cohere2/test_modeling_cohere2.py"
    ["cwm"]="tests/models/cwm/test_modeling_cwm.py"
    ["ernie4_5"]="tests/models/ernie4_5/test_modeling_ernie4_5.py"
    ["exaone4"]="tests/models/exaone4/test_modeling_exaone4.py"
    ["falcon"]="tests/models/falcon/test_modeling_falcon.py"
    ["fsmt"]="tests/models/fsmt/test_modeling_fsmt.py"
    ["gemma"]="tests/models/gemma/test_modeling_gemma.py"
    ["gemma2"]="tests/models/gemma2/test_modeling_gemma2.py"
    ["gemma3"]="tests/models/gemma3/test_modeling_gemma3.py"
    ["gemma3n"]="tests/models/gemma3n/test_modeling_gemma3n.py"
    ["glm"]="tests/models/glm/test_modeling_glm.py"
    ["glm4"]="tests/models/glm4/test_modeling_glm4.py"
    ["gpt2"]="tests/models/gpt2/test_modeling_gpt2.py"
    ["gpt_bigcode"]="tests/models/gpt_bigcode/test_modeling_gpt_bigcode.py"
    ["gpt_neo"]="tests/models/gpt_neo/test_modeling_gpt_neo.py"
    ["gpt_neox"]="tests/models/gpt_neox/test_modeling_gpt_neox.py"
    ["gpt_neox_japanese"]="tests/models/gpt_neox_japanese/test_modeling_gpt_neox_japanese.py"
    ["gptj"]="tests/models/gptj/test_modeling_gptj.py"
    ["helium"]="tests/models/helium/test_modeling_helium.py"
    ["hunyuan_v1_dense"]="tests/models/hunyuan_v1_dense/test_modeling_hunyuan_v1_dense.py"
    ["jais2"]="tests/models/jais2/test_modeling_jais2.py"
    ["led"]="tests/models/led/test_modeling_led.py"
    ["lfm2"]="tests/models/lfm2/test_modeling_lfm2.py"
    ["llama"]="tests/models/llama/test_modeling_llama.py"
    ["longt5"]="tests/models/longt5/test_modeling_longt5.py"
    ["m2m_100"]="tests/models/m2m_100/test_modeling_m2m_100.py"
    ["mamba"]="tests/models/mamba/test_modeling_mamba.py"
    ["mamba2"]="tests/models/mamba2/test_modeling_mamba2.py"
    ["marian"]="tests/models/marian/test_modeling_marian.py"
    ["mbart"]="tests/models/mbart/test_modeling_mbart.py"
    ["ministral"]="tests/models/ministral/test_modeling_ministral.py"
    ["ministral3"]="tests/models/ministral3/test_modeling_ministral3.py"
    ["mistral"]="tests/models/mistral/test_modeling_mistral.py"
    ["mistral3"]="tests/models/mistral3/test_modeling_mistral3.py"
    ["modernbert_decoder"]="tests/models/modernbert_decoder/test_modeling_modernbert_decoder.py"
    ["mpt"]="tests/models/mpt/test_modeling_mpt.py"
    ["mvp"]="tests/models/mvp/test_modeling_mvp.py"
    ["nanochat"]="tests/models/nanochat/test_modeling_nanochat.py"
    ["nemotron"]="tests/models/nemotron/test_modeling_nemotron.py"
    ["olmo"]="tests/models/olmo/test_modeling_olmo.py"
    ["olmo2"]="tests/models/olmo2/test_modeling_olmo2.py"
    ["olmo3"]="tests/models/olmo3/test_modeling_olmo3.py"
    ["opt"]="tests/models/opt/test_modeling_opt.py"
    ["pegasus"]="tests/models/pegasus/test_modeling_pegasus.py"
    ["pegasus_x"]="tests/models/pegasus_x/test_modeling_pegasus_x.py"
    ["persimmon"]="tests/models/persimmon/test_modeling_persimmon.py"
    ["phi"]="tests/models/phi/test_modeling_phi.py"
    ["phi3"]="tests/models/phi3/test_modeling_phi3.py"
    ["plbart"]="tests/models/plbart/test_modeling_plbart.py"
    ["prophetnet"]="tests/models/prophetnet/test_modeling_prophetnet.py"
    ["qwen2"]="tests/models/qwen2/test_modeling_qwen2.py"
    ["qwen3"]="tests/models/qwen3/test_modeling_qwen3.py"
    ["recurrent_gemma"]="tests/models/recurrent_gemma/test_modeling_recurrent_gemma.py"
    ["rwkv"]="tests/models/rwkv/test_modeling_rwkv.py"
    ["seed_oss"]="tests/models/seed_oss/test_modeling_seed_oss.py"
    ["smollm3"]="tests/models/smollm3/test_modeling_smollm3.py"
    ["stablelm"]="tests/models/stablelm/test_modeling_stablelm.py"
    ["starcoder2"]="tests/models/starcoder2/test_modeling_starcoder2.py"
    ["t5"]="tests/models/t5/test_modeling_t5.py"
    ["t5gemma"]="tests/models/t5gemma/test_modeling_t5gemma.py"
    ["t5gemma2"]="tests/models/t5gemma2/test_modeling_t5gemma2.py"
    ["umt5"]="tests/models/umt5/test_modeling_umt5.py"
    ["vaultgemma"]="tests/models/vaultgemma/test_modeling_vaultgemma.py"
    ["xglm"]="tests/models/xglm/test_modeling_xglm.py"
    ["xlstm"]="tests/models/xlstm/test_modeling_xlstm.py"
    ["youtu"]="tests/models/youtu/test_modeling_youtu.py"
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
echo "  Dense Models TP Test Script"
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
