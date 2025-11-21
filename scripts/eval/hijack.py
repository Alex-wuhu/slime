import json
from datasets import load_dataset

# 1. 加载原始数据
ds = load_dataset("gsm8k", "main", split="train")
output_file = "/root/gsm8k_short_cot.jsonl"

print(f"Filtering SHORT CoT data to {output_file}...")

count = 0
with open(output_file, 'w', encoding='utf-8') as f:
    for item in ds:
        question = item['question']
        answer = item['answer']
        
        # 🎯 筛选逻辑：只留答案长度小于 300 字符的题
        # 这种题通常能在 200-400 Token 内做完，600 绝对够用
        if len(answer) > 300:
            continue
            
        # 🎯 温和的 Prompt：允许思考，但别太啰嗦
        # 这比之前的 Hijack 要好，因为它保留了 CoT 的训练价值
        prompt_content = question + "\nPlease reason step-by-step but concisely, and end with #### <Answer>."
        
        new_record = {
            "messages": [{"role": "user", "content": prompt_content}],
            "label": answer
        }
        
        f.write(json.dumps(new_record) + "\n")
        count += 1

print(f"Saved {count} samples. (Dropped long questions to save VRAM)")