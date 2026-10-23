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
- **Byte-level BPE tokenizer** (`tokenizer.ggml.model = "gpt2"`): GPT-2 splitting, isolated digits for
  SmolLM/StarCoder-style vocabularies, special tokens written literally in prompts.
- **Llama-architecture transformer**: grouped-query attention, RoPE, SwiGLU, tied output projection, KV cache.
- **CPU backend**: batched prompt processing, AVX2 Q8_0 kernels (with a scalar fallback), fused Q/K/V and
  gate/up passes, context-split attention across cores.
- **CUDA backend** (`--gpu`): loads the NVIDIA driver and NVRTC at runtime and compiles its kernels for the
  installed GPU on startup. No CUDA SDK is needed to build, and the same binary still runs on machines
  without an NVIDIA GPU. Prompts use an int8 `dp4a` tiled matmul and a FlashAttention-style kernel;
  generation uses f32 activations and split-context ("flash-decoding") attention.
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

- An NVIDIA GPU with compute capability 6.1 or newer (needs `dp4a`), and a current driver (`nvcuda.dll`).
- NVRTC from a CUDA toolkit. betterllama picks the newest `nvrtc64_*.dll` under
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin`, or the one named by `BETTERLLAMA_NVRTC`.
- Windows only for now (the libraries are loaded with `LoadLibraryExW`).

## Supported models

Llama-architecture GGUF files (`general.architecture = "llama"`) with Q8_0 weights and F32/F16 norms, a
byte-level BPE tokenizer and, for `chat`, a ChatML template. Tested with
[SmolLM2-135M-Instruct-Q8_0](https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF). On the CPU,
F16/F32 weight matrices also work. K-quants (Q4_K_M etc.) and other architectures (Qwen3, Gemma, …) are
not supported yet.

## Performance

SmolLM2-135M-Instruct Q8_0, best of 3, tokens per second. Ryzen 5 3600 (6 threads, DDR4-2133) and
RTX 2060 Super, compared with Ollama 0.22.1 on the same GGUF file and the same prompts.

**GPU**

| | betterllama | Ollama |
|---|---|---|
| Prompt, 33 tokens | **3,865** | 2,788 |
| Prompt, 253 tokens | 13,701 | **14,039** |
| Prompt, 1,013 tokens | 12,308 | **15,772** |
| Generate, short context | **286** | 168 |
| Generate at 253 context | **252** | 166 |
| Generate at 1,013 context | **243** | 155 |

**CPU**

| | betterllama | Ollama |
|---|---|---|
| Prompt, 33 tokens | 664 | **802** |
| Prompt, 253 tokens | 692 | **713** |
| Prompt, 1,013 tokens | **518** | 463 |
| Generate, short context | 94 | **130** |
| Generate at 253 context | 84 | **120** |
| Generate at 1,013 context | 63 | **87** |

With a model this small, per-token overhead dominates; larger models are bound by memory bandwidth and
will compare differently.

## Accuracy

The CPU fast path and the GPU prompt path quantise activations to int8 (as llama.cpp does for Q8_0
weights); the GPU generation path uses f32 activations.

| Path | Distance from exact f32 maths (log-probability) |
|---|---|
| GPU generation | 0.0000 at 36–1,566 tokens of context |
| GPU prompt processing | ≤ 0.06 |
| CPU | ≤ 0.10 |

The most likely next token was the same as the exact result in every test.

Tools for checking this:

| | |
|---|---|
| `BETTERLLAMA_EXACT=1` | CPU reference mode: skip activation quantisation (slow) |
| `BETTERLLAMA_STEPWISE=1` | feed prompts one token at a time, to test the generation path |
| `BETTERLLAMA_PROFILE=1` | time per operation (CPU and GPU) in `bench` |
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
| `src/quant.rs` | Q8_0 / f16 formats, AVX2 and scalar matmul kernels |
| `src/model.rs` | CPU transformer and KV cache |
| `src/cuda.rs` | CUDA backend: runtime loading, kernel source, GPU transformer |
| `src/sampler.rs` | greedy / temperature / top-k / top-p sampling |
| `src/profile.rs` | opt-in timing |
| `src/main.rs` | CLI and chat formatting |
