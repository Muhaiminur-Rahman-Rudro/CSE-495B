# Model Download Guide

## Quick Download Commands

### Download Single Model
```bash
python download_models.py --models qwen2.5-3b
```

### Download Multiple Models
```bash
python download_models.py --models qwen2.5-3b mistral-7b
```

### Download All Models
```bash
python download_models.py --all
```

### List Cached Models
```bash
python download_models.py --list
```

## Before Downloading

### 1. Check Disk Space
```bash
df -h /content    # On Colab
df -h ~           # On local
```

**Required Space:**
- Qwen2.5-3B: ~6GB
- Mistral-7B: ~14GB
- LLaMA-2-7B: ~13GB
- LLaMA-2-13B: ~26GB
- Mixtral-8x7B: ~90GB

### 2. HuggingFace Authentication (for LLaMA models)
```bash
# Get token from: https://huggingface.co/settings/tokens
huggingface-cli login

# Request access for LLaMA models:
# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

## Google Colab Free Setup

### Option 1: Download in a Colab Cell
```bash
python download_models.py --models qwen2.5-3b --cache-dir /content/hf_cache
```

### Option 2: Save Cache to Google Drive (Persistent)
```bash
# After mounting Drive in Colab
python download_models.py --models qwen2.5-3b --cache-dir /content/drive/MyDrive/hf_cache
```

### Option 3: Lightweight First, Expand Later
```bash
# Start with one model for free-tier sessions
python download_models.py --models qwen2.5-3b
```

## Local Mac Setup

### Download to Default Cache
```bash
python download_models.py --models qwen2.5-3b
# Downloads to: ~/.cache/huggingface/
```

### Download to External Drive
```bash
python download_models.py --models qwen2.5-3b --cache-dir /Volumes/MyDrive/models
```

## Troubleshooting

### Authentication Errors
```bash
❌ 401 Unauthorized
Solution: 
  1. huggingface-cli login
  2. Request access: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
```

### Disk Space Issues
```bash
⚠️  Only 10GB free, need 14GB
Solution:
  - Free up space
  - Use --cache-dir to specify different location
  - Download models individually instead of --all
```

### Interrupted Downloads
```bash
# Resume download with --force flag
python download_models.py --models qwen2.5-3b --force
```

### Slow Downloads
```bash
# Check internet speed
speedtest-cli

# Download in background
nohup python download_models.py --models qwen2.5-3b > download.log 2>&1 &
```

## Verify Downloads

```bash
# List cached models
python download_models.py --list

# Expected output:
# 📂 Cached Models:
#   ✓ mistralai/Mistral-7B-Instruct-v0.2
#     Size: 14.32 GB
#   ✓ meta-llama/Llama-2-7b-chat-hf
#     Size: 13.48 GB
# Total cache size: 27.80 GB
```

## Cache Management

### Delete Specific Model
```bash
# Find cache location
python download_models.py --list

# Remove model
rm -rf ~/.cache/huggingface/models--mistralai--Mistral-7B-Instruct-v0.2
```

### Clear All Cache
```bash
rm -rf ~/.cache/huggingface/hub/*
```

### Move Cache Location
```bash
# Download to new location
python download_models.py --models qwen2.5-3b --cache-dir /new/path

# Update environment variable
export HF_HOME=/new/path
export TRANSFORMERS_CACHE=/new/path
```
