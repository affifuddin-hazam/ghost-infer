# Benchmarks

[Back to the README](../README.md)

Measured with [GuideLLM](https://github.com/vllm-project/guidellm), vLLM's benchmarking tool, against
[llama.cpp](https://github.com/ggml-org/llama.cpp) and [Ollama](https://ollama.com) on the same GGUF file.

## Test machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 SUPER, 8 GB (Turing, sm_75), driver 577.00 |
| CPU | AMD Ryzen 5 3600, 6 cores / 12 threads |
| RAM | 16 GB DDR4-2133 (2 × 8 GB) |
| Storage | the 4B and 8B on a USB 3.0 external hard drive (read once, at load); the 30B on an NVMe SSD (part of it is read again from the disk while it runs) |
| OS | Windows 10 Pro 22H2 (build 19045), nothing else running (11.7–12.5 GB of RAM free at the start) |
| Engines | ghost-infer (this repo), llama.cpp b11747 (`llama-server` and `llama-bench`, CUDA 12.4), Ollama 0.40.13 |
| Models | Qwen3 8B, Q4_K_M, 5.0 GB (a JOSIEFIED fine-tune: same architecture and size as the base model, so the same speed); Qwen3 4B, Q4_K_M, 2.5 GB; Qwen3-30B-A3B-Instruct-2507, Q3_K_M, 13.7 GB |

## Method

- GuideLLM 0.8.1, `synchronous` profile: one request at a time, the way one person uses a local model.
- Synthetic prompts from the model's own tokenizer, streamed through `/v1/chat/completions`.
- Each server runs alone on the GPU with the whole model on it: ghost-infer `serve --gpu`, llama.cpp
  `-ngl 99 -c 8192`, Ollama with its defaults (all 37 layers on the GPU, 4,096-token context, flash
  attention on auto). Each uses its default KV cache: int8 for ghost-infer, f16 for llama.cpp and Ollama.
- llama.cpp is shown twice. With its defaults, `llama-server` serves 4 users at once and copies every
  prompt's cache to RAM, which costs time on each request; "1 user" adds `-np 1 --cache-ram 0`, its best
  setup for one person.
- Two rounds over every engine, averaged, so GPU clock drift does not favour one of them. 21 measured
  requests per round for the chat workload and 10 for the long one, after a warm-up.
- Reproduce both tables with [`scripts/bench-readme.ps1`](../scripts/bench-readme.ps1) (it runs
  [`scripts/bench-engines.ps1`](../scripts/bench-engines.ps1) for this part); the raw GuideLLM results are
  JSON files, summarised by [`scripts/bench-compare.py`](../scripts/bench-compare.py).

**Chat**: 256-token prompt, 128-token reply

| | ghost-infer | llama.cpp (1 user) | llama.cpp (defaults) | Ollama |
|---|---|---|---|---|
| Generation, tokens/s | **70.3** | 64.8 | 64.7 | 66.0 |
| Prompt reading, tokens/s | 905 | **1,366** | 1,257 | 1,002 |
| Time to first token, ms | 292 | **193** | 210 | 264 |
| Whole request, s | **2.11** | 2.16 | 2.17 | 2.19 |

**Long prompt** (pasted code, a document): 2,048-token prompt, 256-token reply

| | ghost-infer | llama.cpp (1 user) | llama.cpp (defaults) | Ollama |
|---|---|---|---|---|
| Generation, tokens/s | 61.3 | 61.6 | 61.6 | **62.1** |
| Prompt reading, tokens/s | **1,490** | 1,455 | 1,343 | 1,327 |
| Time to first token, ms | **1,380** | 1,413 | 1,531 | 1,549 |
| Whole request, s | 5.56 | **5.55** | 5.67 | 5.65 |

**Generation** is the steady speed while the reply streams (1,000 / mean time between tokens).
**Prompt reading** is prompt tokens divided by the time to the first token, so it includes each server's
own overhead. llama.cpp reads short prompts faster; ghost-infer generates short replies 6–8% faster; on
long prompts all four are within 2% of each other. (Against llama.cpp b9106 from May 2026, ghost-infer
generated 25% faster; llama.cpp has caught up since on models that fit in the GPU.)

## Long contexts

Generation speed of 128 tokens after a prompt already in the cache, without the server:
`ghost-infer bench --gpu --prompt-len <n> -n 128 --no-lookup` (best of 3) against
`llama-bench -ngl 99 -fa 1 -p 0 -n 128 -d <n> -r 2` (llama.cpp's first column is depth 0).

| Generation, tokens/s, after | 256 | 2,000 | 4,000 | 8,000 |
|---|---|---|---|---|
| Qwen3 4B Q4_K_M, ghost-infer | **118.2** | **111.6** | **98.5** | 80.4 |
| Qwen3 4B Q4_K_M, llama.cpp | 107.5 | 101.5 | 94.8 | **84.0** |
| Qwen3 8B Q4_K_M, ghost-infer | **73.3** | **70.6** | **65.1** | 56.5 |
| Qwen3 8B Q4_K_M, llama.cpp | 67.1 | 63.8 | 61.1 | 56.3 |

ghost-infer is 9–11% faster up to 2,000 positions and 4–7% faster at 4,000; at 8,000 the two are level on
the 8B, and llama.cpp is 4% faster on the 4B.

## Mixture of experts bigger than the GPU

Qwen3-30B-A3B-Instruct-2507 Q3_K_M (13.7 GB) on the same 8 GB card with 16 GB of RAM, read from an NVMe
SSD. ghost-infer keeps attention and the router on the GPU, the experts in RAM, and about a third of them
(5.1 GB) in a VRAM cache: the ones picked most recently, since consecutive tokens reuse many experts.
The CPU runs the others, and a background thread copies them into the cache as they are used. llama.cpp
runs it the way its users do: every layer on the GPU, with the experts of some layers left in RAM and
computed on the CPU (`--n-cpu-moe`; a sweep from 16 to 48 layers picked 28). Ollama fits the model the
same way by itself (here the experts of 32 of the 49 layers in RAM, 5.5 GB on the GPU).

Seven varied prompts (writing code, an explanation, a code review with 2,194 tokens of pasted code, a
document to summarise, a chat, a word problem, SQL), each through the engine's `/v1/chat/completions`,
streamed, greedy, 256 tokens; a warm-up round, then two measured rounds
([`scripts/bench-moe.py`](../scripts/bench-moe.py); `bench-readme.ps1 -SkipDense -Stage <folder on an SSD>`):

| Generation, tokens/s | ghost-infer | llama.cpp b11747 (`--n-cpu-moe 28`) | Ollama 0.40.13 |
|---|---|---|---|
| Median of the 14 measured replies | **38.0** | 23.7 | 21.7 |
| Range | 34.5–45.5 | 23.5–23.9 | 21.4–21.9 |
| Writing code / explaining / SQL | **40 / 38 / 38** | 24 | 22 |
| Code review (2,194-token paste) | **36** | 24 | 22 |
| Summarising a document | **45** | 24 | 22 |
| Chat / word problem | **45** / **35** | 24 / 24 | 22 / 22 |

Reading the 2,194-token paste: ghost-infer 414–418 tokens/s, llama.cpp 409–416, Ollama 441–450. Short
prompts (42–72 tokens) reach their first token sooner in llama.cpp and Ollama, 0.6–0.8 s against
ghost-infer's 0.9–1.1 s: such a prompt already touches most of the experts, and ghost-infer runs them all
on the GPU, copying in the ones its cache does not hold.

llama.cpp's and Ollama's speed hardly depends on the text, since each expert stays where it was put.
ghost-infer's does a little: a reply that keeps reaching for new experts (here the word problem) runs
slower than one that stays with a few (the chat, the summary). `ghost-infer bench` shows higher numbers
still: its prompt is one sentence repeated, which keeps the same few experts busy, the cache's best case
rather than a real chat.

## Speculative decoding

Qwen3 8B with `--draft Qwen3-0.6B-Q8_0.gguf` (greedy; the output is identical to decoding without the
draft). The gain depends on how often the small model guesses right:

| Greedy generation, tokens/s | ghost-infer | + draft | guesses accepted |
|---|---|---|---|
| "Write a Python function that checks whether a number is prime…" | 69.0 | **106.0** | 81% |
| "Name three effects of the printing press." | 69.1 | **80.7** | 55% |
| "Explain in simple words why the sky is blue." | 68.7 | **70.1** | 43% |

<details>
<summary>More measurements: startup, small models, CPU</summary>

**Startup** to ready on the GPU with the file in the operating system's cache: Qwen3 8B Q4_K_M 1.9 s
(2.3 s with the 0.6B draft), SmolLM2-135M 0.3 s. A first read from disk takes as long as the disk needs:
about 30 s for the 8B from a USB hard drive.

The tables below use `ghost-infer bench` and `ollama run --verbose`, best of 3, tokens per second.

**SmolLM2-135M-Instruct Q8_0, GPU**

| | ghost-infer | Ollama 0.22.1 |
|---|---|---|
| Prompt, 33 tokens | **5,311** | 2,788 |
| Prompt, 253 tokens | **18,505** | 14,039 |
| Prompt, 1,013 tokens | 14,944 | **15,772** |
| Generate, short context | **277** | 168 |
| Generate at 1,013 context | **243** | 155 |

**Qwen3 8B Q4_K_M, CPU only** (Ryzen 5 3600, 6 threads)

| | ghost-infer | Ollama 0.22.1 |
|---|---|---|
| Prompt, ~260 tokens | 9.4 | **18.2** |
| Generate | 5.2 | **5.9** |

The CPU path works but is not the focus: llama.cpp's CPU kernels are still ahead.

</details>
