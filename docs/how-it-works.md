# How it works

[Back to the README](../README.md)


- **GGUF over a memory map**: weights are never copied on the CPU; on the GPU they stream in through
  pinned buffers, read by all cores straight from the file.
- **CUDA without the SDK**: the driver API and NVRTC are loaded at runtime and the kernels are compiled
  for the installed GPU, so the same binary runs on machines without an NVIDIA GPU.
- **Prompts** use int8 tensor cores (`mma.sync` + `ldmatrix`, sm_75+; a `dp4a` tiled matmul on older
  GPUs) and FlashAttention-2-style attention on f16 tensor cores with f32 accumulation. K-quant prompt
  matmuls keep their epilogue in integers (one activation scale per 128 values).
- **Generation** uses `dp4a` matrix-vector kernels (84–97% of DRAM bandwidth on an 8B model), fused
  RMSNorm/SiLU kernels that write int8 activations directly, split-context ("flash-decoding") attention,
  and a CUDA graph replayed for every one-token step.
- **int8 KV cache** with one scale per position and head, written as each row is produced: about half
  the memory of f16, so an 8B model on an 8 GB card gets a 19,456-token context instead of 10,240.
- **Mixture of experts**: attention and the router on the GPU, experts in RAM plus a VRAM cache of the
  experts picked most recently. An expert the CPU had to run is copied in over the least recently picked
  one, up to 12 a token, by a background thread on its own CUDA stream, which also reads the replaced
  expert back into RAM before the CPU needs it. While generating, cached experts run on the GPU as the CPU
  computes the rest, all of a layer's at once on a pool of spinning threads; prompt batches group
  tokens per expert and stream uncached experts through double-buffered pinned memory. When the model
  is bigger than the free RAM, a prefetch thread reads each layer's experts in as large sequential reads
  one layer ahead (a cold 1,000-token prompt from a USB hard drive: 151 s instead of 215 s), and the chat
  page shows that it is loading weights from disk. Experts copied into GPU memory give their RAM back, and
  short prompts run a token at a time so the GPU-held experts are never read from RAM again: with 16 GB
  of RAM, a new chat's first token came after 20 s instead of 56 s, and replies ran at 14 tokens/s instead of 3.
  The KV cache holds only the positions a conversation has reached, in 4,096-token steps, one buffer per
  layer: the memory a long `--ctx` would reserve holds experts until the conversation needs it, and a
  step's memory comes from the expert cache's last blocks of slots (the other experts stay cached). With
  `--ctx 32768`, GLM-4.7-Flash's short replies ran at 21.7 tokens/s instead of 18.4.
- **Qwen3-Next** (Qwen3-Coder-Next, 80B with 3B active): 36 of its 48 layers are Gated DeltaNet, a recurrent
  kind of attention whose per-head state is decayed, corrected and read once per token (one GPU thread per state
  column, its 128 floats in registers), so only the 12 gated-attention layers keep a KV cache. A recurrent state
  cannot be cut back to an earlier position like a cache, so the end of every prompt is kept as a snapshot, and
  a conversation that goes on from it resumes there. Its 2-bit IQ2 weights stay 2-bit on the GPU (the
  experts in the expert cache too), the embedding table stays in the file, and a prompt runs in passes of
  4,096 tokens, so a long prompt streams the experts that do not fit in VRAM or RAM once. On an RTX 2060
  Super with 16 GB of RAM (IQ2_S, 22 GB), against llama.cpp b11747 on the same prompts: code at 11.8
  tokens/s (9.3), a long answer at 14.5 (12.4), and a 2,867-token prompt in about 77 s (167 s).
- **Qwen3.5 / Qwen3.6** (35B with 3B active): the same design at 40 layers, its value heads taking the key
  heads in turn (the order llama.cpp's converter writes) and its multi-section RoPE, for text, Qwen3-Next's
  partial RoPE. Small enough to stay in RAM: Qwen3.6-35B-A3B (i1-Q3_K_M, 16.8 GB) on the same machine, against
  llama.cpp b11747: code at 31.9 tokens/s (26.0), a long answer at 31.9 (24.9), a 4,027-token prompt in about
  6 s (22 s), and the desktop app's fix-a-failing-test task in 46 s, where Qwen3-Coder-Next took 10 minutes.
- **Multi-head latent attention** (GLM-4.7-Flash, DeepSeek-style): each token caches one compressed
  key-value vector and a shared RoPE key, 54 KB per token where Qwen3-30B-A3B needs 96 KB. Each head's
  query moves into that compressed space and attention runs over the cache as it is, every head scored
  against a row read once; generation splits the context into slices across the GPU. Its experts are
  picked by sigmoid scores with a per-expert bias, beside an always-on shared expert.
- **Speculative decoding** checks all drafted tokens in one batched pass; a token's logits are
  bit-identical whether computed alone or inside a check batch, so the output equals plain greedy output.
- **CPU backend**: AVX2 integer kernels for every format (with a scalar fallback), fused passes and
  context-split attention across cores.

## Accuracy

Like llama.cpp, the fast paths quantise activations to int8. Distance from exact f32 maths, as the
difference in next-token log-probability:

| Path | Distance |
|---|---|
| GPU generation, Q8_0 (SmolLM2) | 0.0000 at 36–1,566 tokens of context |
| GPU generation, Q4_K_M (Qwen3 8B) | ≤ 0.04 |
| GPU prompt processing, Q4_K_M (Qwen3 8B), 40–1,000 tokens | ≤ 0.03 on the top token, ≤ 0.22 over the top 10 |
| int8 KV cache vs f16 KV cache (Qwen3 8B) | ≤ 0.11, same top token at every tested length |
| CPU | ≤ 0.10 |

The most likely next token matched the exact result in every test. Full-vocabulary KL divergence against
exact maths is measured by [`scripts/kld.py`](../scripts/kld.py).

Qwen3-Coder-Next (IQ2_S) was checked against llama.cpp b11747: the same prompt token ids, and the same top 10
next tokens in the same order on the CPU and the GPU (on a prompt with two likely answers, 0.410 / 0.366 on the
CPU and 0.396 / 0.362 on the GPU with an f16 cache, against llama.cpp's 0.408 / 0.367).

Qwen3.6-35B-A3B (i1-Q3_K_M) was checked the same way, on code, English and Malay: the same prompt token ids and
the same five most likely next tokens, their probabilities no further from llama.cpp's than llama.cpp's own
backends are from each other (" George": 0.489 on the CPU, 0.458 on the GPU and 0.481 with exact maths, against
llama.cpp's 0.483 with the experts on the CPU and 0.427 on the CPU alone).

GLM-4.7-Flash (i1-Q3_K_M) was checked against llama.cpp b9106 instead: identical token ids on 12 texts
(code, Chinese, Malay, emoji), the chat template identical to llama.cpp's rendering of the model's own
template (tools included), and the same most likely next token on 7 of 7 prompts of up to 629 tokens on
the CPU and the GPU. The largest log-probability difference in the top 10 was 0.55 (CPU) and 0.69 (GPU);
llama.cpp's own CPU and GPU builds differ by up to 0.85 on the same prompts.

<details>
<summary>Environment variables and accuracy tools</summary>

| | |
|---|---|
| `GHOST_INFER_EXACT=1` | reference mode: no activation quantisation, on the CPU and the GPU (slow) |
| `GHOST_INFER_STEPWISE=1` | feed prompts one token at a time, to test the generation path |
| `GHOST_INFER_PROFILE=1` | time per operation (CPU and GPU) in `bench` |
| `GHOST_INFER_QCH=256` | K-quant GPU prompts: one activation scale per 256 values (~1% faster, about twice the drift) |
| `GHOST_INFER_VERIFY_N=n` | `bench --gpu`: also time one n-token decode batch (a speculative check step) |
| `GHOST_INFER_LOAD_TIMES=1` | GPU: time each startup phase (driver, kernels, weights) |
| `GHOST_INFER_NO_KERNEL_CACHE=1` | GPU: compile the kernels instead of using (and filling) the on-disk cache |
| `GHOST_INFER_NO_GRAPH=1` | GPU: plain kernel launches instead of CUDA-graph replay for one-token steps |
| `GHOST_INFER_NO_TC_ATT=1` | GPU prompts: f32 attention instead of f16 tensor cores (more exact, much slower on long prompts) |
| `GHOST_INFER_NO_Q8K=1` | K-quant GPU prompts: the older per-32 activation-scale kernel (more exact, ~1.7× slower) |
| `GHOST_INFER_EXPERT_CACHE_MB=n` | mixture of experts on the GPU: VRAM for the expert cache (default: what is free; 0 = every expert on the CPU) |
| `GHOST_INFER_FULL_CTX=1` | mixture of experts on the GPU: the whole context's KV cache at load, instead of growing with the conversation |
| `GHOST_INFER_NO_PREFETCH=1` | mixture of experts on the GPU: read expert weights by page faults instead of the prefetch thread's large reads |
| `GHOST_INFER_MOE_CHECK=1` | mixture of experts on the GPU: also compute each GPU-resident expert on the CPU and print the difference |
| `GHOST_INFER_LOGITS_OUT=file` | `top`: write every next-token logit (f32, little-endian) for accuracy tooling |
| `scripts\golden.ps1 [-Save]` | regression check against saved log-probabilities and a greedy reply |
| `scripts\accuracy.ps1 [-Gpu]` | fast path vs exact maths at several prompt lengths |
| `scripts\decode-accuracy.ps1` | GPU generation path vs exact maths at up to ~1,500 tokens |
| `scripts\check_dequant.py` | dequantisation vs gguf-py's reference for every weight format |

</details>
