# betterllama

A small LLM inference engine for GGUF models, written from scratch in Rust. It runs on the CPU, or on an
NVIDIA GPU with `--gpu`, from a single executable with two dependencies (`memmap2`, `rayon`).

```text
> betterllama run SmolLM2-135M-Instruct-Q8_0.gguf -p "What is the capital of France?" --gpu -t 0
[loaded SmolLM2 135M Instruct on GPU NVIDIA GeForce RTX 2060 SUPER (sm_75, NVRTC nvrtc64_120_0.dll) in 0.74s]
The capital of France is Paris. It is a city known for its historical landmarks, cultural institutions,
and cultural attractions. Paris is a major global city, and its population is approximately 2.5 million people.
[prompt 37 tok in 0.01s = 4028.6 tok/s | generated 44 tok in 0.15s = 284.2 tok/s, stopped at end-of-turn]
```

## What it does

- **GGUF v2/v3** reader over a memory map: the model loads in milliseconds and is never copied on the CPU.
- **Byte-level BPE tokenizer** (`tokenizer.ggml.model = "gpt2"`): GPT-2 and Qwen2 splitting, isolated
  digits for SmolLM/StarCoder-style vocabularies, special tokens written literally in prompts.
- **Transformers**: Llama (grouped-query attention, RoPE, SwiGLU, tied or separate output projection) and
  Qwen3 (adds per-head RMSNorm on queries and keys, NEOX-style RoPE), with a KV cache.
- **Weight formats**: Q8_0, Q4_K and Q6_K (so Q4_K_M files work), plus F16/F32 on the CPU.
- **CPU backend**: batched prompt processing, AVX2 integer kernels for every format (with a scalar
  fallback), fused Q/K/V and gate/up passes, context-split attention across cores.
- **CUDA backend** (`--gpu`): loads the NVIDIA driver and NVRTC at runtime and compiles its kernels for the
  installed GPU on startup. No CUDA SDK is needed to build, and the same binary still runs on machines
  without an NVIDIA GPU. Prompts use int8 tensor cores (`mma.sync` + `ldmatrix`, sm_75+; a `dp4a` tiled
  matmul on older GPUs) and FlashAttention-2-style attention on f16 tensor cores (m16n8k8, f32 accumulation;
  an f32 kernel on older GPUs or other head sizes). For K-quant weights the prompt matmul keeps its
  epilogue in integers: activations get one scale per 128 values, the sub-block scales are applied with
  integer multiply-adds and the Q4_K mins go through the tensor cores too. Generation uses `dp4a`
  matrix-vector kernels for K-quants (16-byte loads, 84–97% of DRAM bandwidth on an 8B model), RMSNorm and
  SiLU kernels that write the int8 activations directly, and split-context ("flash-decoding") attention
  that reads each key/value row once per head group.
- **Sampling**: greedy, or temperature + top-k + top-p.
- **Chat**: ChatML template (SmolLM2, Qwen), multi-turn with the KV cache kept between turns.

## Build

```bash
cargo build --release
```

`.cargo/config.toml` builds with `target-cpu=native`, so the binary is tuned for the machine that built it.
Remove that file to build a portable binary.

## Usage

```bash
betterllama run   model.gguf -p "Write a haiku about the ocean."
betterllama chat  model.gguf --gpu
betterllama bench model.gguf --prompt-len 256 -n 128 --gpu
```

| Command | What it does |
|---|---|
| `run <model> -p <prompt>` | one reply, streamed, with timing |
| `chat <model>` | interactive chat (`/reset` clears the conversation, empty line or `/bye` quits) |
| `bench <model>` | fixed workload: `--prompt-len` prompt tokens, then `-n` greedy tokens, best of 3 |
| `top <model> <text> [--gpu]` | the 10 most likely next tokens after raw text, with log-probabilities |
| `tokenize <model> <text>` | token ids and pieces, plus a round-trip check |
| `inspect <model>` | metadata, tensor list and parameter counts |
| `meta <model> <key>` | one metadata value in full (e.g. `tokenizer.chat_template`) |

Options for `run`, `chat` and `bench`:

| Option | Default | |
|---|---|---|
| `-n`, `--max <n>` | 256 | max new tokens per reply |
| `-t`, `--temp <f>` | 0.7 | temperature; `0` = greedy |
| `--top-k <n>` | 40 | keep the k most likely tokens (`0` = all) |
| `--top-p <f>` | 0.9 | nucleus sampling cut-off |
| `--seed <n>` | time | sampling seed |
| `--ctx <n>` | 2048 | context window in tokens |
| `--system <text>` | template's | system prompt |
| `--raw` | | send the prompt as-is, without the chat template |
| `--gpu` | | run on the NVIDIA GPU |
| `--threads <n>` | physical cores | CPU threads (generation is memory-bound, so hyper-threads don't help) |
| `--special` | | print special tokens |

### GPU requirements

- An NVIDIA GPU with compute capability 6.1 or newer (needs `dp4a`; 7.5+ also uses int8 tensor cores),
  and a current driver (`nvcuda.dll`). The weights must fit in VRAM (an 8B Q4_K_M needs about 5.5 GB).
- NVRTC from a CUDA toolkit. betterllama picks the newest `nvrtc64_*.dll` under
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin`, or the one named by `BETTERLLAMA_NVRTC`.
- Windows only for now (the libraries are loaded with `LoadLibraryExW`).

## Supported models

GGUF files with `general.architecture` `llama` or `qwen3`, Q8_0 / Q4_K / Q6_K weights, F32 norms, a
byte-level BPE tokenizer and, for `chat`, a ChatML template. Tested with
[SmolLM2-135M-Instruct-Q8_0](https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF) and a Qwen3 8B
fine-tune in Q4_K_M. Other quantisations (Q4_0, Q5_K, IQ formats, …) and architectures (Gemma, Llama 3's
tokenizer split, …) are not supported yet.

## Performance

Best of 3, tokens per second. Ryzen 5 3600 (6 threads, DDR4-2133) and RTX 2060 Super, compared with
Ollama 0.22.1 on the same GGUF file and the same prompts. Load time is not included.

**Qwen3 8B Q4_K_M, GPU** (both measured in the same session; GPU clocks vary between sessions, so
compare rows only within one table)

| | betterllama | Ollama |
|---|---|---|
| Prompt, ~33 tokens | 745 | **800** |
| Prompt, ~260 tokens | **1,475** | 1,313 |
| Prompt, ~1,040 tokens | **1,477** | 1,390 |
| Generate, short context | **69.7** | 52.7 |
| Generate at ~1,040 context | **65.2** | 52.2 |

**Qwen3 8B Q4_K_M, CPU**

| | betterllama | Ollama |
|---|---|---|
| Prompt, ~33 tokens | 10.3 | **18.7** |
| Prompt, ~260 tokens | 9.4 | **18.2** |
| Generate | 5.2 | **5.9** |

**SmolLM2-135M-Instruct Q8_0, GPU**

| | betterllama | Ollama |
|---|---|---|
| Prompt, 33 tokens | **5,311** | 2,788 |
| Prompt, 253 tokens | **18,505** | 14,039 |
| Prompt, 1,013 tokens | 14,944 | **15,772** |
| Generate, short context | **277** | 168 |
| Generate at 253 context | **254** | 166 |
| Generate at 1,013 context | **243** | 155 |

**SmolLM2-135M-Instruct Q8_0, CPU**

| | betterllama | Ollama |
|---|---|---|
| Prompt, 33 tokens | 664 | **802** |
| Prompt, 253 tokens | 692 | **713** |
| Prompt, 1,013 tokens | **518** | 463 |
| Generate, short context | 94 | **130** |
| Generate at 253 context | 84 | **120** |
| Generate at 1,013 context | 63 | **87** |

On the small model per-token overhead dominates, which betterllama keeps low. On the 8B model the work is
in the matrix kernels, where llama.cpp (inside Ollama) is still ahead, most of all on long prompts.

## Accuracy

Like llama.cpp, the fast paths quantise activations to int8: the CPU everywhere, the GPU for prompts, and
the GPU for generation with K-quant weights. GPU prompt attention runs in f16 (as llama.cpp's does), with
f32 accumulation and f32 key/value cache. GPU generation with Q8_0 weights, and every path under
`BETTERLLAMA_EXACT=1`, uses f32 activations.

| Path | Distance from exact f32 maths (log-probability) |
|---|---|
| GPU generation, Q8_0 (SmolLM2) | 0.0000 at 36–1,566 tokens of context |
| GPU generation, Q4_K_M (Qwen3 8B), 5 prompts | ≤ 0.04 |
| GPU prompt processing, Q4_K_M (Qwen3 8B), 40–1,000 tokens | ≤ 0.03 on the top token, ≤ 0.22 over the top 10 |
| GPU prompt processing, Q8_0 (SmolLM2) | ≤ 0.09 |
| CPU | ≤ 0.10 |

The most likely next token was the same as the exact result in every test. On the Qwen3 8B file it also
matched Ollama (CPU and GPU) on all 5 test prompts, with the same prompt token counts.

Tools for checking this:

| | |
|---|---|
| `BETTERLLAMA_EXACT=1` | reference mode: no activation quantisation, on the CPU and the GPU (slow) |
| `BETTERLLAMA_STEPWISE=1` | feed prompts one token at a time, to test the generation path |
| `BETTERLLAMA_PROFILE=1` | time per operation (CPU and GPU) in `bench` |
| `BETTERLLAMA_QCH=256` | K-quant GPU prompts: one activation scale per 256 values (~1% faster, about twice the drift) |
| `BETTERLLAMA_NO_TC_ATT=1` | GPU prompts: f32 attention instead of f16 tensor cores (more exact, much slower on long prompts) |
| `BETTERLLAMA_NO_Q8K=1` | K-quant GPU prompts: the older per-32 activation-scale kernel (more exact, ~1.7× slower) |
| `scripts\golden.ps1 [-Save]` | regression check against saved log-probabilities and a greedy reply |
| `scripts\accuracy.ps1 [-Gpu]` | fast path vs exact maths at several prompt lengths |
| `scripts\decode-accuracy.ps1` | GPU generation path vs exact maths at up to ~1,500 tokens |

The scripts expect the test model at `models\SmolLM2-135M-Instruct-Q8_0.gguf`.

```bash
cargo test --release
```

## Layout

| File | |
|---|---|
| `src/gguf.rs` | GGUF reader |
| `src/tokenizer.rs` | byte-level BPE tokenizer and UTF-8 streaming |
| `src/quant.rs` | Q8_0 / Q4_K / Q6_K / f16 formats, AVX2 and scalar matmul kernels |
| `src/model.rs` | CPU transformer and KV cache |
| `src/cuda.rs` | CUDA backend: runtime loading, kernel source, GPU transformer |
| `src/sampler.rs` | greedy / temperature / top-k / top-p sampling |
| `src/profile.rs` | opt-in timing |
| `src/main.rs` | CLI and chat formatting |
