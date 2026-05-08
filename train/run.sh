#!/bin/bash
# ============================================================
# 文化感知跨语言安全对齐 — 训练启动脚本
# ============================================================
# 前置条件:
#   1. pip install -r requirements.txt
#   2. 模型下载: python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/mdeberta-v3-base')"
#      (国内机器: export HF_ENDPOINT=https://hf-mirror.com)
#
# 用法:
#   bash train/run.sh                          # 训练全部 8 国
#   bash train/run.sh --countries TH,ID        # 训练指定国家
#   bash train/run.sh --prepare-only           # 仅查看数据统计
#   bash train/run.sh --countries TH --epochs 3 # 测试性训练
#
# GPU 训练: 自动检测 CUDA，无额外配置
# CPU 训练: 自动降低 batch_size 和启用 gradient accumulation
# ============================================================

set -e

# 清除系统代理环境变量 (避免干扰 HuggingFace 下载)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# 跳转到项目根目录
cd "$(dirname "$0")/.."

# 检查依赖
echo "=== 环境检查 ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    import os
    mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1024**3
    print(f'CPU RAM: {mem:.1f} GB')
" 2>&1

echo ""
echo "=== 开始训练 ==="
python -m train "$@"
