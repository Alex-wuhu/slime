import json

# 1. 设置你的输入和输出文件
input_file = "/root/gsm8k_train.jsonl"   # 你现在的训练数据
output_file = "/root/gsm8k_chat_ready.jsonl" # 转换后的文件

print(f"Converting {input_file} to Chat format...")

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    
    for line in fin:
        data = json.loads(line)
        
        # 读取旧的 prompt
        # 如果你的文件里 key 是 "question"，请把下行改为 data['question']
        prompt_text = data.get('prompt') or data.get('question')
        label_text = data.get('label') or data.get('answer')
        
        if not prompt_text:
            continue # 跳过空行

        # 核心：包装成 List[Dict]
        new_record = {
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "label": label_text
        }
        
        fout.write(json.dumps(new_record) + "\n")

print(f"Done! Saved to {output_file}")
print("Key is now: 'messages'")