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
  that reads each key/value row once per head group. One-token steps replay a CUDA graph.
- **Speculative decoding** (`--draft small.gguf`, GPU, greedy): a small model with the same tokenizer
  guesses a few tokens ahead and the main model checks them all in one batched pass. The output is
  exactly the plain greedy output: a token's logits are bit-identical whether it is computed alone or
  inside a check batch, which is tested on whole replies and multi-turn chats.
- **Sampling**: greedy, or temperature + top-k + top-p.
- **Chat**: ChatML template (SmolLM2, Qwen), multi-turn with the KV cache kept between turns.
- **OpenAI-compatible server** (`serve`): `/v1/chat/completions` and `/v1/completions` with streaming,
  `/v1/models`, so chat apps and editor plugins that talk to OpenAI or Ollama can use it. It keeps the
  last conversation in the cache: when a client resends the history with a new message, only the new
  part is read.

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
betterllama run   Qwen3-8B-Q4_K_M.gguf -p "Explain DNS." -t 0 --gpu --draft Qwen3-0.6B-Q8_0.gguf
betterllama serve Qwen3-8B-Q4_K_M.gguf --gpu --port 8080
```

| Command | What it does |
|---|---|
| `run <model> -p <prompt>` | one reply, streamed, with timing |
| `chat <model>` | interactive chat (`/reset` clears the conversation, empty line or `/bye` quits) |
| `serve <model>` | OpenAI-compatible HTTP server (see below) |
| `bench <model>` | fixed workload: `--prompt-len` prompt tokens, then `-n` greedy tokens, best of 3 |
| `top <model> <text> [--gpu]` | the 10 most likely next tokens after raw text, with log-probabilities |
| `tokenize <model> <text>` | token ids and pieces, plus a round-trip check |
| `inspect <model>` | metadata, tensor list and parameter counts |
| `meta <model> <key>` | one metadata value in full (e.g. `tokenizer.chat_template`) |

Options for `run`, `chat`, `serve` and `bench`:

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
| `--draft <gguf>` | | speculative decoding with this small model (`--gpu`, used for greedy replies; same tokenizer needed) |
| `--draft-k <n>` | 3 | tokens the draft proposes per step (1-7) |
| `--host <addr>` | 127.0.0.1 | `serve`: address to listen on (`0.0.0.0` for other machines on your network) |
| `--port <n>` | 8080 | `serve`: port |

### Server

`betterllama serve model.gguf --gpu` listens on `http://127.0.0.1:8080/v1`. Point any OpenAI client at that
base URL (the API key and model name can be anything):

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | `messages` (system / user / assistant, text content), `stream`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`, `stream_options.include_usage` |
| `POST /v1/completions` | the same with a raw `prompt` instead of messages (no chat template) |
| `GET /v1/models` | the loaded model (its file name) |
| `GET /health` | `{"status":"ok"}` |

Requests run one at a time. Unset sampling fields use the command-line defaults (`-t`, `--top-k`,
`--top-p`, `-n`). Greedy requests (`"temperature": 0`) use the `--draft` model when one is loaded, with
the same output. Replies include special tokens such as Qwen3's `<think>` blocks as text, which chat apps
that understand them can fold away. Images, tools/function calls, log-probabilities and several choices
per request (`n`) are not supported.

### GPU requirements

- An NVIDIA GPU with compute capability 6.1 or newer (needs `dp4a`; 7.5+ also uses int8 tensor cores),
  and a current driver (`nvcuda.dll`). The weights must fit in VRAM (an 8B Q4_K_M needs about 5.5 GB).
- NVRTC 11.1 or newer from a CUDA toolkit. betterllama picks the newest `nvrtc64_*.dll` under
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin`, or the one named by `BETTERLLAMA_NVRTC`.
  The compiled kernels are cached in `%LOCALAPPDATA%\betterllama\kernels` (about 0.6 MB), so only the
  first start after an update or a GPU change compiles them (a few seconds).
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

**Startup** to ready on the GPU, with the model file in the operating system's cache (any recent run):
Qwen3 8B Q4_K_M 1.9 s, with the 0.6B draft 2.3 s, SmolLM2-135M 0.3 s. The weights stream to the GPU through
pinned buffers, read by all cores straight from the file. A first read from disk takes as long as the disk
needs: about 30 s for the 8B from a USB hard disk, a few seconds from an SSD.

**Qwen3 8B Q4_K_M, GPU** (both measured in the same session; GPU clocks vary between sessions, so
compare rows only within one table)

| | betterllama | Ollama |
|---|---|---|
| Prompt, ~33 tokens | 745 | **800** |
| Prompt, ~260 tokens | **1,475** | 1,313 |
| Prompt, ~1,040 tokens | **1,477** | 1,390 |
| Generate, short context | **71.4** | 52.7 |
| Generate at ~1,040 context | **67.0** | 52.2 |

**Speculative decoding**, the same 8B with `--draft JOSIEFIED-Qwen3-0.6B-Q8_0.gguf` (k = 3), same session as
Ollama. The gain depends on how often the draft guesses right, so it is shown per text:

| Greedy generation | betterllama | + draft | Ollama | guesses accepted |
|---|---|---|---|---|
| bench text, ~32-token prompt (64 tokens) | 71.4 | **96.4** | 52.7 | |
| bench text, ~256-token prompt | 70.1 | **119.4** | 52.5 | |
| bench text, ~1,024-token prompt | 67.0 | **108.5** | 52.2 | |
| "Name three effects of the printing press." (200 tokens) | 69.1 | **80.7** | 52.0 | 55% |
| "Write a Python function that checks whether a number is primeâ€¦" | 69.0 | **106.0** | 51.8 | 81% |
| "Explain in simple words why the sky is blue." | 68.7 | **70.1** | 51.7 | 43% |

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
| `BETTERLLAMA_VERIFY_N=n` | `bench --gpu`: also time one n-token decode batch (a speculative check step) |
| `BETTERLLAMA_LOAD_TIMES=1` | GPU: time each startup phase (driver, kernels, weights) |
| `BETTERLLAMA_NO_KERNEL_CACHE=1` | GPU: compile the kernels instead of using (and filling) the on-disk cache |
| `BETTERLLAMA_NO_GRAPH=1` | GPU: plain kernel launches instead of CUDA-graph replay for one-token steps |
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
