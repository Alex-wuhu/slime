#!/bin/bash
set -ex

### -------------------------------------------------------
### 1. 先优雅关闭 slime/sclang actors
### -------------------------------------------------------
echo "Stopping previous Slime/SGLang actors..."
pkill -f "sglang.launch_server" || true
pkill -f "train.py" || true
sleep 2

# 在当前 Shell 设置，防止 Ray Head 启动时没吃到

### -------------------------------------------------------
### 2. 启动 Ray
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
        --dashboard-port=9000
else
    echo "Ray already running. Skipping start."
fi

### -------------------------------------------------------
### 3. NVLINK 检测
### -------------------------------------------------------
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

### -------------------------------------------------------
### 4. Model Args
### -------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# 确保这个路径下有 qwen3-4B.sh，如果没有请手动 source 正确的 config
if [ -f "${SCRIPT_DIR}/models/qwen3-4B.sh" ]; then
    source "${SCRIPT_DIR}/models/qwen3-4B.sh"
else
    # 如果找不到配置脚本，这里给一个兜底的默认值，防止报错
    MODEL_ARGS=(
        --model-type qwen
        --seq-length 2048
    )
fi

CKPT_ARGS=(
   --hf-checkpoint /root/Qwen3-4B
   # ❌ [修改1] 注释掉 ref-load，节省显存，让它自动复用 policy model
   --ref-load /root/Qwen3-4B-1117_torch_dist
   # ❌ [修改2] 注释掉 load，防止加载之前的坏档。等跑通一次后，如果需要断点续训再解开。
   --load /root/Qwen3-4B_slime/
   
   --finetune
   --save /root/Qwen3-4B_slime/
   
   # ✅ [修改3] 增加保存间隔，避免频繁 I/O 卡死训练
   --save-interval 50
   --no-save-optim
)

ROLLOUT_ARGS=(
   --prompt-data /root/gsm8k_short_cot.jsonl
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   
   # 保持你的设置：10个采样，每次推理1个，最安全
   --num-rollout 16
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 600
   --rollout-temperature 1.0
   --global-batch-size 4
   --balance-data
)

EVAL_ARGS=(
   # --eval-interval 30  # 稍微拉长一点评估间隔
   # --eval-prompt-data gsm8k /root/gsm8k_test.jsonl
   # --n-samples-per-eval-prompt 1
   # --eval-max-response-len 512
   # --eval-top-p 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 36
   --use-dynamic-batch-size
   --max-tokens-per-gpu 1024
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

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev
   --wandb-group qwen3-4B-test
   --wandb-key 640a67e6a962c9b99965bf69e3a757879675ba76   
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # ✅ 保持这个 0.05，这对省显存非常重要
   --sglang-mem-fraction-static 0.1
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

### -------------------------------------------------------
### 5. Runtime Env
### -------------------------------------------------------
# ✅ [修改4] 将显存碎片优化参数注入到 Ray 的环境中
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING\": \"1\"
  }
}"

### -------------------------------------------------------
### 6. 提交 Slime 任务
### -------------------------------------------------------
RAY_ENABLE_RECORD_ACTOR_TASK_LOGGING=1
echo "Submitting Slime job..."

# 注意：WANDB_ARGS 需要在 MODEL_ARGS 之后
ray job submit --address="http://127.0.0.1:9000" \
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
       ${WANDB_ARGS[@]} \
       ${MISC_ARGS[@]}