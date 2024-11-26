# Auto Evol Instruct

## How to run
```
# 
git clone https://github.com/kkendama/Auto-Evol-Instruct.git
cd Auto-Evol-Instruct

# Install Packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Start LLM server
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --quantization awq --max-model-len 4096 --gpu_memory-utilization 0.95

# Optimize prompt
sh optimize.sh

# Evolve instruction
sh evolve.sh
```