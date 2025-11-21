
pip install -U huggingface_hub

# Download model weights (Qwen3-4B)
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B


# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k


# Download training dataset (openai/gsm8k)
hf download --repo-type dataset openai/gsm8k \
  --local-dir /root/gsm8k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
# Set model arguments for conversion
 source /workspace/slime/scripts/models/qwen3-4B.sh

# Convert Hugging Face checkpoint to Megatron-LM torch distributed format for ref training
PYTHONPATH=/root/Megatron-LM:/workspace/slime \
python /workspace/slime/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B-1117_torch_dist


# Verify the converted model files
ls /root/Qwen3-4B-1117_torch_dist


# Run training script

  bash /workspace/slime/scripts/run-qwen3-4B.sh

# Access the development container
docker exec -it slime-dev bash


# 
PYTHONPATH=/root/Megatron-LM python  /workspace/slime/tools/convert_torch_dist_to_hf.py \
  --input-dir /root/Qwen3-4B_slime/iter_0000009 \
  --output-dir /root/Qwen3-4B_iter_009 \
  --origin-hf-dir /root/Qwen3-4B