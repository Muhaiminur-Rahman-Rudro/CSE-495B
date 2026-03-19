#!/usr/bin/env python3
"""
Download models for offline use on Colab or local development.

Usage:
    python download_models.py --models qwen2.5-3b mistral-7b
    python download_models.py --all
    python download_models.py --models qwen2.5-3b --cache-dir /custom/path
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# Model mapping
MODEL_CONFIGS = {
    "qwen2.5-3b": {
        "id": "Qwen/Qwen2.5-3B-Instruct",
        "size_gb": 6,
    },
    "mistral-7b": {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "size_gb": 14,
    },
    "llama2-7b": {
        "id": "meta-llama/Llama-2-7b-chat-hf",
        "size_gb": 13,
        "requires_auth": True,
    },
    "llama2-13b": {
        "id": "meta-llama/Llama-2-13b-chat-hf",
        "size_gb": 26,
        "requires_auth": True,
    },
    "mixtral-8x7b": {
        "id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "size_gb": 90,
    },
}


def check_disk_space(cache_dir: Path, required_gb: float) -> bool:
    """Check if enough disk space is available."""
    free_bytes = shutil.disk_usage(cache_dir.parent).free
    free_gb = free_bytes / (1024**3)
    
    if free_gb < required_gb + 5:  # Add 5GB buffer
        print(f"⚠️  Warning: Only {free_gb:.1f}GB free, need {required_gb:.1f}GB")
        return False
    return True


def download_model(
    model_name: str,
    cache_dir: Path,
    force: bool = False
) -> bool:
    """Download a single model."""
    
    if model_name not in MODEL_CONFIGS:
        print(f"❌ Unknown model: {model_name}")
        print(f"   Available: {', '.join(MODEL_CONFIGS.keys())}")
        return False
    
    config = MODEL_CONFIGS[model_name]
    model_id = config["id"]
    
    print(f"\n{'='*60}")
    print(f"📦 Downloading: {model_name}")
    print(f"   HuggingFace ID: {model_id}")
    print(f"   Size: ~{config['size_gb']}GB")
    print(f"{'='*60}\n")
    
    # Check auth requirement
    if config.get("requires_auth", False):
        print("🔐 This model requires HuggingFace authentication")
        print("   Make sure you've run: huggingface-cli login")
        print(f"   And have access to: https://huggingface.co/{model_id}\n")
    
    # Check disk space
    if not check_disk_space(cache_dir, config["size_gb"]):
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Check if already cached
    model_cache_name = model_id.replace('/', '--')
    model_cache_path = cache_dir / f"models--{model_cache_name}"
    
    if model_cache_path.exists() and not force:
        print(f"✓ Model already cached at: {model_cache_path}")
        print("  Use --force to re-download")
        return True
    
    try:
        # Download tokenizer
        print("→ Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )
        print("  ✓ Tokenizer downloaded")
        
        # Download model
        print("→ Downloading model weights (this may take 10-20 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=str(cache_dir),
            low_cpu_mem_usage=True,
            device_map="cpu",  # Keep on CPU during download
            trust_remote_code=True,
        )
        print("  ✓ Model downloaded")
        
        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"\n✓ Successfully downloaded {model_name}")
        print(f"  Cache location: {model_cache_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {model_name}")
        print(f"   Error: {e}")
        
        if "401" in str(e) or "403" in str(e):
            print("\n🔑 Authentication Error:")
            print("   1. Login: huggingface-cli login")
            print(f"   2. Request access: https://huggingface.co/{model_id}")
        
        return False


def list_cached_models(cache_dir: Path):
    """List all cached models."""
    print("\n📂 Cached Models:\n")
    
    if not cache_dir.exists():
        print("  No cache directory found")
        return
    
    model_dirs = list(cache_dir.glob("models--*"))
    
    if not model_dirs:
        print("  No models cached yet")
        return
    
    for model_dir in model_dirs:
        model_name = model_dir.name.replace("models--", "").replace("--", "/")
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        
        print(f"  ✓ {model_name}")
        print(f"    Size: {size_gb:.2f} GB")
        print(f"    Path: {model_dir}\n")
    
    # Total cache size
    total_cache = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
    print(f"Total cache size: {total_cache / (1024**3):.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Download LLM models for offline use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific models
    python download_models.py --models qwen2.5-3b mistral-7b
  
  # Download all models
  python download_models.py --all
  
  # Use custom cache directory
    python download_models.py --models qwen2.5-3b --cache-dir /data/models
  
  # List cached models
  python download_models.py --list
  
  # Force re-download
    python download_models.py --models qwen2.5-3b --force
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model(s) to download"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/workspace/cache" if os.path.exists("/workspace") else str(Path.home() / ".cache/huggingface"),
        help="Cache directory for models"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List cached models and exit"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    
    args = parser.parse_args()
    
    # Setup cache directory
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ['HF_HOME'] = str(cache_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
    os.environ['TORCH_HOME'] = str(cache_dir)
    
    print(f"📁 Cache directory: {cache_dir}")
    
    # List cached models
    if args.list:
        list_cached_models(cache_dir)
        return
    
    # Determine which models to download
    if args.all:
        models_to_download = list(MODEL_CONFIGS.keys())
    elif args.models:
        models_to_download = args.models
    else:
        parser.print_help()
        return
    
    print(f"\n🚀 Downloading {len(models_to_download)} model(s)...")
    
    total_size = sum(MODEL_CONFIGS[m]["size_gb"] for m in models_to_download)
    print(f"📊 Total download size: ~{total_size}GB\n")
    
    # Download each model
    success_count = 0
    for model_name in models_to_download:
        if download_model(model_name, cache_dir, args.force):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 Download Summary:")
    print(f"   Successfully downloaded: {success_count}/{len(models_to_download)}")
    print(f"   Cache location: {cache_dir}")
    print(f"{'='*60}\n")
    
    if success_count < len(models_to_download):
        sys.exit(1)


if __name__ == "__main__":
    main()
