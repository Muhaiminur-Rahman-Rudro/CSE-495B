"""
GPU optimization utilities for cloud environments.
"""

import torch
import logging

logger = logging.getLogger(__name__)


def optimize_for_gpu():
    """Apply GPU-specific optimizations."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. GPU optimizations skipped.")
        return

    # Enable TF32 for better performance on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set memory allocation config
    torch.cuda.set_per_process_memory_fraction(0.95)

    # Enable cudnn benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True

    # Set default dtype to bfloat16 if available
    if torch.cuda.is_bf16_supported():
        torch.set_default_dtype(torch.bfloat16)
        logger.info("Using bfloat16 precision")
    else:
        torch.set_default_dtype(torch.float16)
        logger.info("Using float16 precision")

    # Log GPU info
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        logger.info(f"  Compute Capability: {props.major}.{props.minor}")


def get_optimal_batch_size(model_name: str, gpu_memory_gb: float) -> int:
    """Estimate optimal batch size based on GPU memory."""
    # Rough estimates for 7B models
    model_sizes = {
        "qwen2.5-3b": 6,
        "mistral-7b": 14,  # GB for full precision
        "llama2-7b": 14,
        "llama2-13b": 26,
    }

    base_size = model_sizes.get(model_name, 14)
    available = gpu_memory_gb - base_size - 2  # Reserve 2GB for overhead

    if available <= 0:
        return 1

    # Each sample takes roughly 1-2GB
    return max(1, int(available / 2))


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
