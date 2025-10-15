# HOWTO
WMMA port for RDNA3.x (tested for gfx1151 Strix Halo)

## Prereqs
- ROCm + HIP installed (hipcc, rocminfo available).
- Python packages for exporter: pip install numpy safetensors huggingface_hub tqdm
- Optional (if dequantizing HF quantized weights): pip install accelerate transformers

## Checkout
```
git clone https://github.com/lhl/gpt-oss-amd.git
cd gpt-oss-amd
```
- Quick check: `hipcc --version && rocminfo | rg -m1 gfx || true`

## Build
```
./run.sh build  (OMP flavor; builds build/run)

```
- If gfx11/Strix Halo, WMMA path is auto-enabled on detection. To be explicit: GPU_ARCH=gfx1151 make runomp

## Quick Tests (GPU WMMA kernels)
```
GPU_ARCH=gfx1151 ./run.sh test -t 60
```
- Runs: WMMA GEMM + attention unit tests; expect small bf16 errors and mismatches=0.

## Export Model (.bin)
First your probably need to dequant BF16 safetensors (eg MXFP4 models):
```
python3 tools/dequantize_to_bf16.py 20b --src /path/to/hf_snapshot --dst /path/to/gpt-oss-20b-bf16
```
Then you can export from the BF16:
```
./run.sh export --snapshot /path/to/gpt-oss-20b-bf16 -o gpt-oss-20b.bin
```
- Tip: Print available keys: `./run.sh export --model-id openai/gpt-oss-20b -o /tmp/x.bin --print-keys`

## GPU E2E Smoke (getp)
- Short run (blocking enabled by default for stability):
```
./run.sh run -c ./gpt-oss-20b.bin -m getp -n 64 -b 1 -t 4 -f
```
- Defaults input/output to tests/data/{input,output}.txt (creates input if missing).
- Use --no-blocking to disable HIP_LAUNCH_BLOCKING=1.

Generate GT with CPU & Verify (optional, recommended)
- Generate a CPU ground-truth token ID file (greedy):
  - `./run.sh run -c ./gpt-oss-20b.bin -m getp_cpu -i tests/data/input.txt -o tests/gt/output_20b.txt -n 64 -t 4`
- Run GPU and verify against GT:
  - `./run.sh run -c ./gpt-oss-20b.bin -m getp -n 64 -b 1 -t 4 -f -v tests/gt/output_20b.txt`

Decode Output (optional)
- Decode token IDs to text:
  - `./run.sh decode -i tests/data/output.txt -l`
  - Saves to gt_decoded.txt.

## Troubleshooting
- ROCm libs not found: export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
- Slurm available: add FORCE_SRUN=1 to force srun.
- If you see HIP “unspecified launch failure,” keep blocking ON (default) or re-run with HIP_LAUNCH_BLOCKING=1.

## Sample Output

~2 Minutes to generate output verification on CPU:
```
🐟 ❯ ./run.sh run -c ./gpt-oss-20b.bin -m getp_cpu -i tests/data/input.txt -o tests/gt/output_20b.txt -n 64 -t 4
                      __
                     |  \
  ______    ______  _| $$_           ______    _______   _______
 /      \  /      \|   $$ \ ______  /      \  /       \ /       \
|  $$$$$$\|  $$$$$$\$$$$$$|      \|  $$$$$$\|  $$$$$$$$
| $$  | $$| $$  | $$ | $$ __\$$$$$$| $$  | $$ \$$    \  \$$$    \
| $$__| $$| $$__/ $$ | $$|  \      | $$__/ $$ _\$$$$$$\ _\$$$$$$\
 \$$    $$| $$    $$  \$$  $$       \$$    $$|       $$|       $$
 _\$$$$$$$| $$$$$$$$    \$$$$         \$$$$$$  \$$$$$$$  \$$$$$$$
|  \__| $$| $$
 \$$    $$| $$
  \$$$$$$  \$$
  ════════════════════════════════════════════════════════════════
                       gpt-oss-amd from scratch
              https://github.com/tuanlda78202/gpt-oss-amd
  ════════════════════════════════════════════════════════════════

==================================================================
[RUN] 2025-10-15 12:01:12
  cwd           : /home/lhl/gpt-oss-amd
  MODELBIN_ROOT : /gpu_trainee/final-project/modelbin
  checkpoint    : ./gpt-oss-20b.bin
  mode          : getp_cpu (provided)
  gpus(-g)      : 1 (default)
  input(-i)     : tests/data/input.txt (provided)
  output(-o)    : tests/gt/output_20b.txt (provided)
  tokenizer(-z) : <unset>
  system(-y)    : <unset>
  temp(-T)      : 0.0 (run.cpp default)
  top_p(-p)     : 0.9 (run.cpp default)
  steps(-n)     : 64 (provided)
  seed(-s)      : time(NULL) (run.cpp default)
  batch_size(-b): 32 (run.cpp default)
  profiling(-f) :  (disabled)
  logging(-l)   :  (disabled)
  truncate(-t)  : 4 (limit to first 4 lines)
  kv_cache      : bf16 (default)
  blocking      : enabled (HIP_LAUNCH_BLOCKING)
>>> env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib HIP_LAUNCH_BLOCKING=1 build/run "./gpt-oss-20b.bin" -m getp_cpu -i tests/data/input.txt -o tests/gt/output_20b.txt -n 64 -t 4
 EXECUTING: env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib HIP_LAUNCH_BLOCKING=1 build/run "./gpt-oss-20b.bin" -m getp_cpu -i tests/data/input.txt -o tests/gt/output_20b.txt -n 64 -t 4
[run] enter main
📊 PROFILING DISABLED
[run] loading checkpoint: ./gpt-oss-20b.bin
vocab_size: 201088
hidden_dim: 2880
n_experts: 32
experts_per_token: 4
intermediate_dim: 2880
n_layers: 24
head_dim: 64
n_attn_heads: 64
n_kv_heads: 8
max_seq_len: 131072
init context len: 4096
rope theta: 150000.000000
rope_scaling_factor: 32.000000
sliding window: 128
swiglu_limit: 7.000000
[run] mmap ok, file_size=83659028796 bytes
[run] mapping weights...
[run] mapped weights
[run] after load_checkpoint
[run] after malloc_run_state
[run] checkpoint loaded. n_layers=24, hidden_dim=2880
```

About 3 minutes for GPU inference test:
```
🐟 ❯ ./run.sh run -c ./gpt-oss-20b.bin -m getp -n 64 -b 1 -t 4 -f
                      __
                     |  \
  ______    ______  _| $$_           ______    _______   _______
 /      \  /      \|   $$ \ ______  /      \  /       \ /       \
|  $$$$$$\|  $$$$$$\$$$$$$|      \|  $$$$$$\|  $$$$$$$$
| $$  | $$| $$  | $$ | $$ __\$$$$$$| $$  | $$ \$$    \  \$$$    \
| $$__| $$| $$__/ $$ | $$|  \      | $$__/ $$ _\$$$$$$\ _\$$$$$$\
 \$$    $$| $$    $$  \$$  $$       \$$    $$|       $$|       $$
 _\$$$$$$$| $$$$$$$$    \$$$$         \$$$$$$  \$$$$$$$  \$$$$$$$
|  \__| $$| $$
 \$$    $$| $$
  \$$$$$$  \$$
  ════════════════════════════════════════════════════════════════
                       gpt-oss-amd from scratch
              https://github.com/tuanlda78202/gpt-oss-amd
  ════════════════════════════════════════════════════════════════

==================================================================
[RUN] 2025-10-15 12:04:29
  cwd           : /home/lhl/gpt-oss-amd
  MODELBIN_ROOT : /gpu_trainee/final-project/modelbin
  checkpoint    : ./gpt-oss-20b.bin
  mode          : getp (default)
( model)
  gpus(-g)      : 1 (default)
  input(-i)     : tests/data/input.txt (run.sh default for getp)
  output(-o)    : tests/data/output.txt (run.sh default for getp)
  verify(-v)    : tests/gt/output_20b.txt (run.sh default for getp)
  tokenizer(-z) : <unset>
  system(-y)    : <unset>
  temp(-T)      : 0.0 (run.cpp default)
  top_p(-p)     : 0.9 (run.cpp default)
  steps(-n)     : 64 (provided)
  seed(-s)      : time(NULL) (run.cpp default)
  batch_size(-b): 1 (provided)
  profiling(-f) : enabled (forward timing)
  logging(-l)   :  (disabled)
  truncate(-t)  : 4 (limit to first 4 lines)
  kv_cache      : bf16 (default)
  blocking      : enabled (HIP_LAUNCH_BLOCKING)
>>> env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib HIP_LAUNCH_BLOCKING=1 build/run "./gpt-oss-20b.bin" -m getp -i tests/data/input.txt -o tests/data/output.txt -n 64 -b 1 -f 1 -v tests/gt/output_20b.txt -t 4
 EXECUTING: env LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib:/opt/rocm/lib HIP_LAUNCH_BLOCKING=1 build/run "./gpt-oss-20b.bin" -m getp -i tests/data/input.txt -o tests/data/output.txt -n 64 -b 1 -f 1 -v tests/gt/output_20b.txt -t 4
[run] enter main
📊 PROFILING ENABLED
[run] loading checkpoint: ./gpt-oss-20b.bin
vocab_size: 201088
hidden_dim: 2880
n_experts: 32
experts_per_token: 4
intermediate_dim: 2880
n_layers: 24
head_dim: 64
n_attn_heads: 64
n_kv_heads: 8
max_seq_len: 131072
init context len: 4096
rope theta: 150000.000000
rope_scaling_factor: 32.000000
sliding window: 128
swiglu_limit: 7.000000
[run] mmap ok, file_size=83659028796 bytes
[run] mapping weights...
[run] mapped weights
[run] after load_checkpoint
[run] after malloc_run_state
[run] checkpoint loaded. n_layers=24, hidden_dim=2880
requests size = 66560 B
num requests: 4
==================================================================
🔥 WARMING UP...
[Parallel Config] dp=1, ep=1, devices=1 (available=1), batch_size=1
GPU 0 (AMD Radeon Graphics): 120.0 GB free / 120.0 GB total
Using 16-bit KV cache (bfloat16) with cyclic buffers
Converting and transferring weights...
  token_embedding_table (1.1 GB)... done
  out (1.1 GB)... done
✅ Hybrid precision model loaded: 39.1 GB allocated

--- HYBRID WARM-UP COMPLETE (device 0 | dp=0 ep=0) ---
GPU Memory Status: Total 120.00 GB, Used 39.14 GB, Free 80.86 GB
-----------------------------------------------
⌛️ Warm up (s): 49.612000
==================================================================
⚡️ RUNNING INFERENCE...
🚀 [DP device 0] Ready with batch_size = 1
#1: Hello |
#1  ●●●●●●●●●●●●●●●● ✓ Done

┌─────────────────────────────┐
│ ⌛️ Time: 110.591000         │
│ ⚡️ TPS: 2.016439            │
└─────────────────────────────┘
==================================================================
🔍 VERIFYING OUTPUT...
⚠️ Request #1: Length mismatch (GT: 61 tokens, Generated: 60 tokens)
⚠️ Request #2: Length mismatch (GT: 58 tokens, Generated: 57 tokens)
⚠️ Request #3: Length mismatch (GT: 57 tokens, Generated: 56 tokens)
⚠️ Request #4: Length mismatch (GT: 51 tokens, Generated: 50 tokens)

📊 Verification Summary:
Total requests checked: 4
Requests with mismatches: 4
Requests matching: 0
❌ TESTS FAILED! 4 requests have mismatches.
==================================================================
♻️  FREE GPU MEMORY...
GPU memory: 119.9 GB free / 120.0 GB total

--- HYBRID FINISH COMPLETE ---
GPU Memory Status:
  Total: 120.00 GB
  Used: 0.14 GB
  Free: 119.86 GB
-------------------------------
⌛️ Finish (s): 0.247000
❌ Verification: FAILED
```

Original upstream README below:

---

<div align="center">

# GPT-OSS from Scratch on AMD GPUs

 <p>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-lightgrey.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/github/actions/workflow/status/tuanlda78202/gpt-oss-amd/ci.yaml?branch=main&label=CI&logo=github" alt="CI Status">
    <img src="https://img.shields.io/github/last-commit/tuanlda78202/gpt-oss-amd?&label=commit" alt="Last Commit">
 </p>

[Overview](#overview) | [Build & Run](#build-and-run) | [Experiments](#experiments) | [Acknowledgements](#acknowledgments) | [Contributions](#contributions)

<img width="1696" height="980" alt="image" src="https://github.com/user-attachments/assets/efd81a09-5299-4bac-b382-66e576a48b1f" />

</div>

## Overview

After six years-the first time since GPT-2, OpenAI has released new open-weight LLMs, `gpt-oss-20b` and `gpt-oss-120b`. From day one, many inference engines such as llama.cpp, vLLM, and SGLang have supported these models; however, most focus on maximizing throughput using CUDA for NVIDIA GPUs, offering limited support for AMD GPUs. Moreover, their library-oriented implementations are often complex to understand and difficult to adapt for personal or experimental use cases.

To address these limitations, we introduce `gpt-oss-amd`, a pure C++ implementation of OpenAI’s GPT-OSS models designed to **maximize inference throughput on AMD GPUs without relying on external libraries**. Our goal is to explore end-to-end LLM optimization, from kernel-level improvements to system-level design, providing insights for researchers and developers interested in high-performance computing and model-level optimization.

Inspired by [llama2.c](https://github.com/karpathy/llama2.c), our implementation uses HIP (an AMD programming model equivalent to CUDA) and avoids dependencies such as rocBLAS, hipBLAS, RCCL, and MPI. We utilize multiple optimization strategies for the 20B and 120B models, including efficient model loading, batching, multi-streaming, multi-GPU communication, optimized CPU–GPU–SRAM memory access, FlashAttention, matrix-core–based GEMM, and load balancing for MoE routing. Experiments on a single node with 8× AMD MI250 GPUs show that our implementation achieves over 30k TPS on the 20B model and nearly 10k TPS on the 120B model in custom benchmarks, demonstrating the effectiveness of our optimizations and the strong potential of AMD GPUs for large-scale LLM inference.

---

## Roadmap

- [x] Release codebase
- [ ] Publish worklog blog post

## Build and Run

### Code Structure

```plain
gpt-oss-amd/
   ├── include/              # Header files
   ├── src/
   │   ├── getp/             # Request serving and runtime logic
   │   ├── hip/              # Custom HIP kernels for AMD GPUs
   │   ├── forward.cpp       # Model forward pass implementation
   ├── tests/                # Evaluation scripts
   ├── tools/                # Model/tokenizer conversion and HF inference utilities
   └── run.sh                # Build and run script
```

### Resources

- Download GPT-OSS 20/120B model `safetensors` files from [here](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) and convert them to `bin` using the provided script in `tools/model_export` to can use with the C++ inference runtime.

- Tokenizer compatible with OpenAI `o200k_harmony` (via `tiktoken`).

- GCC/Clang with OpenMP and HIP/ROCm installed.

### Setup Env

```bash
uv sync
source .venv/bin/activate
pre-commit install
chmod +x run.sh
```

### Build

```bash
./run build [default|fast|omp]
```

### Run

- Chat

  ```bash
  # interactive turn-based generation, optional system prompt
  ./run run -m chat -i "How do I tune top-p?" -y "You are a concise assistant." -T 0.7
  ```

- Single-Prompt

  ```bash
  # single prompt → completion
  ./run run -m generate -i "Write a haiku about parallelism." -T 0.8 -p 0.95
  ```

- Batch

  ```bash
  # multi-prompt batch
  ./run run                          # default 20B, 1 GPU, uses tests/data/{input,output}.txt
  ./run run -m 120 -g 8 --kv16       # 120B, 8 GPUs, KV 16-bit
  ```

### Help

```bash
# full, colorized usage summary
./run.sh -h
```

## Experiments

| Model          | Mode   | Num Requests | Num GPUs     | Warm-up (s) | Throughput (TPS) | METEOR | BERTScore |
| -------------- | ------ | ------------ | ------------ | ----------- | ---------------- | ------ | --------- |
| `gpt-oss-20b`  | `getp` | 7120         | 8x AMD MI250 | 20          | 30086            | 0.52   | 0.98      |
| `gpt-oss-120b` | `getp` | 6144         | 8x AMD MI250 | 46          | 9993             | 0.55   | 0.99      |

---

## Acknowledgments

This project was part of the GPU Engineer Training Program, a collaboration between [Moreh](https://www.linkedin.com/company/moreh-vietnam/) and [THUNDER Research Group](http://snuvm.snu.ac.kr/) (Seoul National University).

## License

MIT License — free to use, adapt, and share for learning and working.

## Contributions

Found a bug, typo, or want to extend something? Open a PR — all contributions are welcome.

<p align="left">
  <a href="https://github.com/tuanlda78202/gpt-oss-amd/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=tuanlda78202/gpt-oss-amd" />
  </a>
</p>
