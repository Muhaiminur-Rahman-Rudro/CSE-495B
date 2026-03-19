# Google Colab Free Quick Start

This guide provides the primary Google Colab Free workflow.

## 1. Create a New Colab Notebook

1. Go to https://colab.research.google.com
2. Create a new notebook.
3. In Runtime -> Change runtime type, select:
- Hardware accelerator: GPU
- Runtime shape: Standard (free)

## 2. Setup Environment

Run this cell first:

```python
!git clone <your-repo-url> CSE-495B_NLP-main
%cd CSE-495B_NLP-main
!pip install -U pip
!pip install -r requirements.txt
```

## 3. Authenticate HuggingFace (Optional but Recommended)

Needed for gated models such as LLaMA.

```python
from huggingface_hub import notebook_login
notebook_login()
```

## 4. Set Colab Cache Paths

Use Colab local storage for speed.

```python
import os

os.environ["HF_HOME"] = "/content/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/content/hf_cache"
os.environ["TORCH_HOME"] = "/content/hf_cache"
```

## 5. Optional: Mount Google Drive for Persistent Results

Use this if you want results to survive runtime resets.

```python
from google.colab import drive
drive.mount('/content/drive')
```

If mounted, prefer writing outputs to a Drive path, for example:
- `/content/drive/MyDrive/CSE495B_results`

## 6. Quick Validation Run

```python
!python -m experiments.run_experiment \
  --model qwen2.5-3b \
  --prompting cot \
  --decoding greedy \
  --dataset gsm8k \
  --max-samples 10 \
  --output-dir /content/results_quick
```

## 7. Colab Free Benchmark Preset

Use the Colab-friendly config:

```python
!python -m experiments.run_experiment --config experiments/configs/colab_free.yaml
```

## 8. Keep Runs Lightweight on Colab Free

- Use one model at a time.
- Keep `max_samples` low (20-100).
- Prefer `quantization: "4bit"`.
- Reduce `max_new_tokens` to 256-384.
- Avoid large multi-strategy sweeps in one session.

## 9. Save Outputs to Drive (Optional)

```python
!cp -r results /content/drive/MyDrive/CSE495B_results
```

## 10. Troubleshooting

### CUDA out of memory

Reduce these in config:
- `max_new_tokens`
- `num_beams`
- `tot_num_branches`
- `tog_num_hypotheses`

### Runtime disconnected

- Save outputs to Google Drive after each run.
- Re-run setup cells and continue with smaller batches.

### Slow model download

- Start with `qwen2.5-3b` only.
- Skip pre-downloading multiple models on free-tier runtimes.
