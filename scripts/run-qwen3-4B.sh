#!/bin/bash
set -ex

### -------------------------------------------------------
### 1. 先优雅关闭 slime/sclang actors（不要杀 python / ray）
### -------------------------------------------------------
echo "Stopping previous Slime/SGLang actors..."
pkill -f "sglang.launch_server" || true
pkill -f "train.py" || true
sleep 2


### -------------------------------------------------------
### 2. 启动 Ray（如果没启动）
### -------------------------------------------------------
if ! pgrep -f "raylet" >/dev/null; then
    echo "Starting Ray head..."
    export CUDA_VISIBLE_DEVICES=0
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

    ray start --head \
        --node-ip-address ${MASTER_ADDR} \
        --num-gpus 1 \
        --disable-usage-stats \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265
else
    echo "Ray already running. Skipping start."
fi


### -------------------------------------------------------
### 3. NVLINK 检测代码（保持不变）
### -------------------------------------------------------
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"


### -------------------------------------------------------
### 4. Load Model Args（保持不变）
### -------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-4B.sh"


CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   --ref-load /root/Qwen3-4B-1117_torch_dist
   --load /root/Qwen3-4B_slime/
   --save /root/Qwen3-4B_slime/
   --save-interval 10 
)

ROLLOUT_ARGS=(
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 50
   --rollout-batch-size 4
   --n-samples-per-prompt 3
   --rollout-max-response-len 512
   --rollout-temperature 0.8
   --global-batch-size 12
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 1024
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 0
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
   --attention-backend flash
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.02
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.05
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)


### -------------------------------------------------------
### 5. Runtime Env（保持）
### -------------------------------------------------------
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"


### -------------------------------------------------------
### 6. 提交 Slime 任务（不需要 stop ray）
### -------------------------------------------------------
echo "Submitting Slime job..."
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 /workspace/slime/train.py \
       --actor-num-nodes 1 \
       --actor-num-gpus-per-node 1 \
       --colocate \
       ${MODEL_ARGS[@]} \
       ${CKPT_ARGS[@]} \
       ${ROLLOUT_ARGS[@]} \
       ${OPTIMIZER_ARGS[@]} \
       ${GRPO_ARGS[@]} \
       ${PERF_ARGS[@]} \
       ${EVAL_ARGS[@]} \
       ${SGLANG_ARGS[@]} \
       ${MISC_ARGS[@]}
