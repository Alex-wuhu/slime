import subprocess
import time
import requests
import signal
import os


## python /workspace/slime/scripts/eval/inference_compare.py 
MEM_FRACTION = "0.35" 

MODELS = [
    {"name": "BASE", "path": "/root/Qwen3-4B", "port": 31000},
    {"name": "FINETUNE", "path": "/root/Qwen3-4B_iter_009", "port": 31001}
]
PROMPT = """"content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\".",
"role": "user" 
"""


def kill_all():
    print("🧹 清理旧进程...")
    # 确保杀干净
    os.system("ps -ef | grep sglang | grep -v grep | awk '{print $2}' | xargs -r kill -9")
    time.sleep(2)

def wait_ready(port, name, log_file, timeout=120):
    url = f"http://127.0.0.1:{port}/health"
    print(f"⏳ 等待 {name} (Port {port}) 就绪...", end="", flush=True)
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # 检查日志是否报错退出
        if os.path.exists(log_file.name):
            with open(log_file.name, 'r') as f:
                # 读取最后 1000 字节检查是否有 RuntimeError
                try:
                    f.seek(0, 2)
                    size = f.tell()
                    f.seek(max(size - 1024, 0))
                    tail = f.read()
                    if "RuntimeError" in tail or "Error:" in tail:
                        print(f"\n❌ {name} 启动报错，请检查日志！")
                        return False
                except:
                    pass

        try:
            if requests.get(url, timeout=1).status_code == 200:
                print(" ✅ 就绪")
                return True
        except:
            pass
        time.sleep(1)
        print(".", end="", flush=True)
    
    print(f" ❌ 超时！请查看日志文件: {log_file.name}")
    return False

def launch_one(model_config):
    log_filename = f"server_{model_config['name'].lower()}.log"
    print(f"\n🚀 启动 {model_config['name']} (日志: {log_filename})...")
    
    f = open(log_filename, "w")
    
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_config['path'],
        "--port", str(model_config['port']),
        "--mem-fraction-static", MEM_FRACTION,
        "--trust-remote-code",
        "--host", "0.0.0.0",
        # === 修改点 2: 限制上下文长度，防止 OOM ===
        "--context-length", "8192",
        # === 保持禁用图优化，加快启动 ===
        "--disable-cuda-graph"
    ]
    
    p = subprocess.Popen(cmd, stdout=f, stderr=f, preexec_fn=os.setsid)
    return p, f

def main():
    kill_all()
    
    running_procs = []
    open_files = []
    
    try:
        # 1. 启动 BASE
        for m in MODELS:
            proc, log_file = launch_one(m)
            running_procs.append(proc)
            open_files.append(log_file)
            
            if not wait_ready(m['port'], m['name'], log_file):
                print(f"⚠️ {m['name']} 启动失败，脚本停止。")
                return

        # 2. 推理
        print("\n" + "="*20 + " 开始推理测试 " + "="*20)
        for m in MODELS:
            print(f"\n🔹 模型: {m['name']}")
            payload = {
                "text": PROMPT,
                "sampling_params": {"temperature": 0.2, "max_new_tokens": 256}
            }
            try:
                r = requests.post(f"http://127.0.0.1:{m['port']}/generate", json=payload, timeout=120)
                if r.status_code == 200:
                    print(f"✅ 输出:\n{r.json()['text']}")
                else:
                    print(f"❌ 错误: {r.text}")
            except Exception as e:
                print(f"❌ 请求异常: {e}")

    finally:
        print("\n🛑 停止服务...")
        for p in running_procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except:
                pass
        for f in open_files:
            f.close()
        kill_all()

if __name__ == "__main__":
    main()