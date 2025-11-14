
pip install -U huggingface_hub

# Download model weights (Qwen3-4B)
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B


# Download training dataset (dapo-math-17k)
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Download evaluation dataset (aime-2024)
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024


  docker exec -it slime-dev bash