
pip install -U huggingface_hub

# Download model weights (Qwen3-4B)
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B


# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024

# Set model arguments for conversion
source /root/slime/scripts/models/qwen3_4B.sh

# Convert Hugging Face checkpoint to Megatron-LM torch distributed format for ref training
PYTHONPATH=/root/Megatron-LM python /workspace/slime/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B-1117_torch_dist


  docker exec -it slime-dev bash