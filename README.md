<p align="center">
  <img src="docs/ghost.png" width="132" alt="The ghost-infer ghost">
</p>

<h1 align="center">ghost-infer</h1>

<p align="center">
  <b>Fast local LLM inference for the GPU you already have.</b><br>
  One small Rust program that runs GGUF models on an NVIDIA card, serves an OpenAI-compatible API,<br>
  and comes with a desktop app where the model works on your files, and a chat page where a little ghost<br>
  shows you what the model is doing.
</p>

<p align="center">
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#the-chat-page">Chat page</a> ·
  <a href="#the-desktop-app-and-agent">Desktop app</a> ·
  <a href="#server-api">Server API</a> ·
  <a href="#how-it-works">How it works</a>
</p>

<p align="center">
  <img src="docs/demo.gif" width="860" alt="The ghost-infer desktop app finding and fixing a bug, then running the tests">
  <br><sub>Real time, not sped up: the desktop app fixing a failing test, Qwen3-30B-A3B-Instruct-2507 Q3_K_M on an RTX 2060 SUPER (8 GB) with 16 GB of RAM.</sub>
</p>

## Why ghost-infer

Most inference engines are built for data centres: many users, big GPUs, batching. ghost-infer is built
for the other case: **one person, one consumer GPU, one model at a time**, and it is tuned for exactly that.

On an 8 GB RTX 2060 SUPER with 16 GB of RAM, against current llama.cpp (b11747) and Ollama (0.40.13) on
the same GGUF files:

| Generation speed | ghost-infer | llama.cpp | Ollama |
|---|---|---|---|
| Qwen3 8B, 5 GB: fits in the GPU (128-token reply) | **70.3 tokens/s** | 64.8 tokens/s | 66.0 tokens/s |
| Qwen3-30B-A3B, 13.7 GB: bigger than the GPU (varied prompts) | **38.0 tokens/s** (34–46) | 23.7 tokens/s | 21.7 tokens/s |

On a model that fits in the GPU, ghost-infer is 6–8% faster on short replies and level on long prompts;
llama.cpp generates faster once the context passes about 4,000 tokens. On a mixture-of-experts model
bigger than the GPU, ghost-infer generates 60% faster than llama.cpp's best setting, though it takes longer
to the first token of a short prompt. Details and method in [Benchmarks](#benchmarks).

- **One file, nothing to install.** A ~2 MB executable with two Rust dependencies (`memmap2`, `rayon`).
  No CUDA SDK, no Python, no Docker: it loads the NVIDIA driver at runtime and compiles its kernels for
  your GPU on first start.
- **Bigger models than your GPU.** Mixture-of-experts models keep their experts in RAM and the ones a
  reply is using in a VRAM cache: Qwen3-30B-A3B runs at about 38 tokens/s on an 8 GB card with 16 GB of RAM.
- **Drop-in server.** OpenAI-compatible `/v1/chat/completions` with streaming and tool calling, so editor
  plugins, chat apps and agent harnesses work unchanged. Switch models from the chat page or the API.
- **Exact where it matters.** Speculative decoding gives bit-identical output to plain greedy decoding,
  dequantisation matches the GGUF reference bit for bit, and accuracy is tracked against exact f32 maths.

## Quick start

```bash
git clone https://github.com/affifuddin-hazam/ghost-infer
cd ghost-infer
cargo build --release
```

Put some GGUF files in a folder (for example `models\`) and start the server on one of them:

```bash
target\release\ghost-infer serve models\Qwen3-8B-Q4_K_M.gguf --gpu
```

Open <http://127.0.0.1:8080> to chat, or point any OpenAI client at `http://127.0.0.1:8080/v1`. The
other GGUF files in the same folder appear in the chat page's model picker.

Requirements: Windows 10/11, an NVIDIA GPU with compute capability 6.1 or newer (GTX 10-series and
later), a current driver, and the CUDA toolkit's NVRTC library (see [GPU requirements](#gpu-requirements)).
Without `--gpu` it runs on the CPU (AVX2).

## Benchmarks

Measured with [GuideLLM](https://github.com/vllm-project/guidellm), vLLM's benchmarking tool, against
[llama.cpp](https://github.com/ggml-org/llama.cpp) and [Ollama](https://ollama.com) on the same GGUF file.

### Test machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 2060 SUPER, 8 GB (Turing, sm_75), driver 577.00 |
| CPU | AMD Ryzen 5 3600, 6 cores / 12 threads |
| RAM | 16 GB DDR4-2133 (2 × 8 GB) |
| Storage | the 8B on a USB 3.0 external hard drive; the 30B on an NVMe SSD (part of it is read again from the disk while it runs) |
| OS | Windows 10 Pro 22H2 (build 19045), nothing else running (11.7–12.5 GB of RAM free at the start) |
| Engines | ghost-infer (this repo), llama.cpp b11747 (`llama-server` and `llama-bench`, CUDA 12.4), Ollama 0.40.13 |
| Models | Qwen3 8B, Q4_K_M, 5.0 GB (a JOSIEFIED fine-tune: same architecture and size as the base model, so the same speed); Qwen3 4B, Q4_K_M, 2.5 GB; Qwen3-30B-A3B-Instruct-2507, Q3_K_M, 13.7 GB |

### Method

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
- Reproduce both tables with [`scripts/bench-readme.ps1`](scripts/bench-readme.ps1) (it runs
  [`scripts/bench-engines.ps1`](scripts/bench-engines.ps1) for this part); the raw GuideLLM results are
  JSON files, summarised by [`scripts/bench-compare.py`](scripts/bench-compare.py).

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

### Long contexts

Generation speed of 128 tokens after a prompt already in the cache, without the server:
`ghost-infer bench --gpu --prompt-len <n> -n 128 --no-lookup` (best of 3) against
`llama-bench -ngl 99 -fa 1 -p 0 -n 128 -d <n> -r 2` (llama.cpp's first column is depth 0).

| Generation, tokens/s, after | 256 | 2,000 | 4,000 | 8,000 |
|---|---|---|---|---|
| Qwen3 4B Q4_K_M, ghost-infer | **117.4** | **106.2** | 91.1 | 73.6 |
| Qwen3 4B Q4_K_M, llama.cpp | 107.3 | 99.7 | **93.9** | **83.1** |
| Qwen3 8B Q4_K_M, ghost-infer | **73.1** | **68.5** | **61.8** | 53.1 |
| Qwen3 8B Q4_K_M, llama.cpp | 66.2 | 63.1 | 60.6 | **56.0** |

ghost-infer is 6–10% faster up to 2,000 positions; from about 4,000, llama.cpp's attention pulls ahead,
and at 8,000 it is 5% (8B) to 13% (4B) faster.

### Mixture of experts bigger than the GPU

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
([`scripts/bench-moe.py`](scripts/bench-moe.py); `bench-readme.ps1 -SkipDense -Stage <folder on an SSD>`):

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

### Speculative decoding

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

## The chat page

`ghost-infer serve` includes a chat page, compiled into the program and working offline:

<p align="center">
  <img src="docs/screenshot-desktop.png" width="860" alt="The chat page: a conversation with code, the ghost on the message box">
</p>

- **A ghost that shows what the model is doing**: it listens while you type, thinks while the model reads
  the prompt or reasons in `<think>`, talks while the reply streams, reaches for a magnifier on tool calls,
  and changes face with the model's mood (the model is asked to tag its replies; the tags stay hidden).
- **Built for coding too**: code blocks with their language and a copy button, tables, copy and
  write-again on every reply, and a reply that keeps streaming while you scroll back to read.
- **Saved conversations** in the browser, grouped by day.
- **Model picker**: lists the GGUF files next to the running model; choosing one swaps it on the GPU.
- **Thinking switch** for reasoning models, and settings for temperature, reply length and system prompt.
- **A tool to check this computer**: the model can call `get_system_info` for the CPU, RAM, GPU, drives and
  what is running, and the reply shows what it looked at. It only reads; a setting turns it off.
- Live speed while a reply streams, and per-reply timing under each answer.
- Works on a phone too: start the server with `--host 0.0.0.0` and open it from another device on your network.

<p align="center">
  <img src="docs/screenshot-mobile.png" width="260" alt="The chat page on a phone">
</p>

## The desktop app and agent

`ghost-app` is a native window onto the server, drawn on the CPU so it takes no GPU memory from the model.
The server runs the agent: the model works on your files and your computer with tools, round after
round, asks before anything that changes something, and keeps every chat in `%LOCALAPPDATA%\ghost-infer`.

<p align="center">
  <img src="docs/screenshot-app.png" width="860" alt="The desktop app: a bug found and fixed, its plan done, and the rail with the model, the context, the tasks, the changed file and the scheduled tasks">
</p>

```bash
cargo build --release -p ghost-app
target\release\ghost-infer serve models\Qwen3-8B-Q4_K_M.gguf --gpu
target\release\ghost-app                # connects to 127.0.0.1:8080; give another host:port to change it
```

The app runs on the same computer as the server: the agent only answers requests from this computer.

- **Tools, with your OK.** The + menu turns tool groups on: reading files (on by default), changing files,
  running PowerShell commands, the web, and MCP tools. File tools only work inside the folders you add.
  Changing files, running commands, the web, MCP tools and installing a skill ask first (once, always, or
  no), and every reply's file changes can be undone.
- **Plans and checks.** The model keeps a task list, pinned above the message box. After changing code it
  is sent back once to run it. With **Keep going until done** on (+ menu), a reply that stops with steps
  of its list still open is sent back to them.
- **Long chats fit.** Before each round, old tool outputs are taken out, then the oldest turns, so a chat
  never outgrows the context; the chat file keeps everything, and the server's cache stays valid.
- **Instructions and memory.** `GHOST.md` goes into every chat: the one in `%LOCALAPPDATA%\ghost-infer`,
  and one in each folder you add (a folder's `CLAUDE.md` or `AGENTS.md` works too). The model saves
  lasting facts with `remember` and can search earlier chats.
- **Skills and MCP servers.** Install [Agent Skills](https://agentskills.io) (a folder with a `SKILL.md`)
  from a folder, a .zip or a GitHub link; a repository with several skills lets you pick. MCP servers go in
  `mcp.json` in Claude Desktop's format, and **From Claude** copies the ones Claude Desktop has.
- **Scheduled tasks.** Ask in a chat ("remind me in 20 minutes to…", "every weekday at 9:00, check the
  build"): the model schedules it once, every N minutes or daily at a time, and it runs in that chat even
  while you are away. Its reply waits there, marked unread.
- **The heartbeat.** Turn it on in the rail (every 15 minutes to 2 hours) and write a checklist in
  `HEARTBEAT.md`: the model goes through it and tells you only when something needs you. A reply of
  `HEARTBEAT_OK` is dropped, and a checklist with nothing in it is not checked at all.

Scheduled tasks and the heartbeat wait until nothing else has used the model for 20 seconds, and use only
the tool groups you set to always allowed, since nobody is there to ask.

| In `%LOCALAPPDATA%\ghost-infer` | |
|---|---|
| `chats\` | every chat, for the app and any other client of the agent |
| `harness.json` | the folders tools may use, and the groups always allowed |
| `GHOST.md`, `memory.json` | instructions and remembered facts, for every chat |
| `skills\`, `mcp.json` | installed skills and MCP servers |
| `schedules.json`, `HEARTBEAT.md` | scheduled tasks, and the heartbeat and its checklist |
| `app.json` | the app's own settings |

## Usage

```bash
ghost-infer serve model.gguf --gpu                  # server + chat page on http://127.0.0.1:8080
ghost-infer chat  model.gguf --gpu                  # chat in the terminal
ghost-infer run   model.gguf -p "Explain DNS." --gpu
ghost-infer bench model.gguf --prompt-len 256 -n 128 --gpu
ghost-infer run   Qwen3-8B-Q4_K_M.gguf -p "Explain DNS." -t 0 --gpu --draft Qwen3-0.6B-Q8_0.gguf
```

| Command | What it does |
|---|---|
| `serve <model>` | OpenAI-compatible HTTP server with the chat page |
| `chat <model>` | interactive chat (`/reset` clears the conversation, empty line or `/bye` quits) |
| `run <model> -p <prompt>` | one reply, streamed, with timing |
| `bench <model>` | fixed workload: `--prompt-len` prompt tokens, then `-n` greedy tokens, best of 3 |
| `top <model> <text> [--gpu]` | the 10 most likely next tokens after raw text, with log-probabilities |
| `tokenize <model> <text>` | token ids and pieces, plus a round-trip check |
| `inspect <model>` | metadata, tensor list and parameter counts |
| `meta <model> <key>` | one metadata value in full (e.g. `tokenizer.chat_template`) |
| `dequant <model> <tensor> <row>` | one row of a weight tensor as floats (for checking formats) |

<details>
<summary>Options for <code>run</code>, <code>chat</code>, <code>serve</code> and <code>bench</code></summary>

| Option | Default | |
|---|---|---|
| `--gpu` | | run on the NVIDIA GPU |
| `-n`, `--max <n>` | 256 | max new tokens per reply |
| `-t`, `--temp <f>` | 0.7 | temperature; `0` = greedy |
| `--top-k <n>` | 40 | keep the k most likely tokens (`0` = all) |
| `--top-p <f>` | 0.9 | nucleus sampling cut-off |
| `--seed <n>` | time | sampling seed |
| `--ctx <n>` | `run`, `bench`: 2048; `chat`, `serve`: as much as fits | context window in tokens. On the GPU `chat` and `serve` take the longest that fits in the free memory, keeping 384 MB spare (on Windows, going over does not fail: the driver quietly moves memory to system RAM and everything slows down); on the CPU 4096 |
| `--kv <f16\|int8>` | int8 (`GHOST_INFER_KV`) | GPU KV cache format: int8 holds about twice the context of f16 |
| `--draft <gguf>` | | speculative decoding with this small model (`--gpu`, greedy replies; same tokenizer needed) |
| `--draft-k <n>` | 3 | tokens the draft proposes per step (1-7) |
| `--system <text>` | template's | system prompt |
| `--no-think` | | reasoning models answer without thinking first (requests can still ask for it) |
| `--raw` | | send the prompt as-is, without the chat template |
| `--threads <n>` | physical cores | CPU threads (generation is memory-bound, so hyper-threads don't help) |
| `--special` | | print special tokens |
| `--host <addr>` | 127.0.0.1 | `serve`: address to listen on (`0.0.0.0` for other machines on your network) |
| `--port <n>` | 8080 | `serve`: port |
| `--models <dir>` | | `serve`: another folder of GGUF files for the model picker, besides the model's own (repeatable) |

</details>

## Server API

Point any OpenAI client at `http://127.0.0.1:8080/v1` (the API key and model name can be anything):

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

| Endpoint | |
|---|---|
| `POST /v1/chat/completions` | `messages` (system / user / assistant / tool, text content), `tools`, `tool_choice` (`"none"` hides the tools), `stream`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stop`, `stream_options.include_usage`, `chat_template_kwargs.enable_thinking` |
| `POST /v1/completions` | the same with a raw `prompt` instead of messages (no chat template) |
| `GET /v1/models` | the loaded model (its file name) and its `context_length` |
| `GET /api/models` | the GGUF files in the model's folder, the current model, and one being loaded |
| `POST /api/models/load` | `{"id": "<file name without .gguf>"}`: switch to that model; answers once it is ready |
| `GET /api/system` | this computer: CPU, RAM, GPU (VRAM, temperature, load), OS, drives, and the model and engine running (the chat page's `get_system_info` tool) |
| `GET /api/progress` | the prompt being read: `done` and `total` tokens (`total` 0: none), and `disk`: waiting for weights to come off the disk |
| `GET /` | the chat page |
| `GET /health` | `{"status":"ok"}` |

- Requests run one at a time. Unset sampling fields use the command-line defaults.
- The server keeps the last conversation in its cache: when a client resends the history with one new
  message, only the new part is read.
- Greedy requests (`"temperature": 0`) use the `--draft` model when one is loaded, with the same output.
- **Tool calling** works the OpenAI way (`tools`, `tool_calls`, `role: "tool"` results). Tools are written
  into the prompt the way Qwen's chat template does, and the model's `<tool_call>` blocks come back as
  `tool_calls` with `finish_reason: "tool_calls"`; streamed, each call is one `delta.tool_calls` chunk.
- **Switching models** drops the running model (freeing its GPU memory) and loads the new one. Requests
  during a switch get `503`; if the new model fails to load, the server goes back to the previous one.
- The agent's endpoints (`/api/agent`, `/api/tools`, `/api/chats` and the rest the app uses) are not part
  of this API: they answer only requests from this computer that carry the token written into the chat
  page. Other agent harnesses bring their own loop and tools and use `/v1`.
- Not supported: images, log-probabilities, several choices per request (`n`).

## Supported models

| | |
|---|---|
| Architectures | Llama, Qwen3, Qwen3-MoE, and DeepSeek-style models such as GLM-4.7-Flash (`general.architecture` `llama`, `qwen3`, `qwen3moe`, `deepseek2`; on the GPU, `deepseek2` with a 512-wide latent and up to 32 heads) |
| Weight formats | Q8_0, Q3_K, Q4_K, Q5_K, Q6_K (so Q3_K_M, Q4_K_M, Q5_K_M, Q6_K and Q8_0 files), F16/F32 on the CPU; Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL and BF16 by dequantising, for the few tensors llama.cpp stores that way |
| Tokenizer | byte-level BPE (`gpt2`): GPT-2 / Qwen2 / GLM-4 splitting, SmolLM-style digits |
| Chat template | ChatML (SmolLM2, Qwen) with Qwen3's thinking and tool-call format; GLM-4's format with its XML tool calls |
| Tested with | [SmolLM2-135M-Instruct](https://huggingface.co/unsloth/SmolLM2-135M-Instruct-GGUF), Qwen3 0.6B and 8B fine-tunes, [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF), [GLM-4.7-Flash (Huihui abliterated, i1-Q3_K_M)](https://huggingface.co/mradermacher/Huihui-GLM-4.7-Flash-abliterated-i1-GGUF) |

Not yet: IQ formats other than IQ4_NL, Gemma, Llama 3's tokenizer split, images.

## GPU requirements

- An NVIDIA GPU with compute capability 6.1 or newer (needs `dp4a`; 7.5+ also uses int8 tensor cores),
  and a current driver. Dense models must fit in VRAM (an 8B Q4_K_M needs about 5.5 GB); mixture-of-experts
  models only need their shared weights to fit.
- NVRTC 11.1 or newer from a [CUDA toolkit](https://developer.nvidia.com/cuda-downloads). ghost-infer
  picks the newest `nvrtc64_*.dll` under `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*\bin`, or
  the one named by `GHOST_INFER_NVRTC`. Compiled kernels are cached in `%LOCALAPPDATA%\ghost-infer\kernels`
  (about 0.6 MB), so only the first start after an update or a GPU change compiles them.
- Windows only for now (Linux is on the roadmap).

`.cargo/config.toml` builds with `target-cpu=native`, so the binary is tuned for the machine that built
it. Remove that file to build a portable binary.

## How it works

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
exact maths is measured by [`scripts/kld.py`](scripts/kld.py).

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
| `GHOST_INFER_NO_PREFETCH=1` | mixture of experts on the GPU: read expert weights by page faults instead of the prefetch thread's large reads |
| `GHOST_INFER_MOE_CHECK=1` | mixture of experts on the GPU: also compute each GPU-resident expert on the CPU and print the difference |
| `GHOST_INFER_LOGITS_OUT=file` | `top`: write every next-token logit (f32, little-endian) for accuracy tooling |
| `scripts\golden.ps1 [-Save]` | regression check against saved log-probabilities and a greedy reply |
| `scripts\accuracy.ps1 [-Gpu]` | fast path vs exact maths at several prompt lengths |
| `scripts\decode-accuracy.ps1` | GPU generation path vs exact maths at up to ~1,500 tokens |
| `scripts\check_dequant.py` | dequantisation vs gguf-py's reference for every weight format |

</details>

## Roadmap

- Linux support
- More architectures (Gemma, Llama 3 tokenizer) and formats (Q4_0, IQ quants)
- Prebuilt releases
- Constrained decoding for tool calls

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work
by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional
terms or conditions.

The chat page includes [three.js](https://threejs.org) (MIT, [`src/vendor/three.LICENSE`](src/vendor/three.LICENSE))
and the [Instrument Sans](https://github.com/Instrument/instrument-sans) typeface (SIL Open Font License 1.1,
[`src/vendor/instrument-sans.LICENSE`](src/vendor/instrument-sans.LICENSE)).
