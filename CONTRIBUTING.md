# Contributing to ghost-infer

Thanks for helping. ghost-infer is a small project with one maintainer, so a few notes up front save
everyone time.

## Before you start

- **Bugs**: open an issue with the bug report form. The GPU, driver, model file and exact command matter
  more than anything else: most problems only happen with one model or one card.
- **A model that doesn't load**: use the model request form and link the GGUF file.
- **Bigger changes** (a new architecture, a new backend, a change to the server API): open an issue first
  so we can agree on the shape before you write a lot of code.
- **Security problems**: don't open a public issue; see [SECURITY.md](SECURITY.md).

Small fixes (typos, docs, an obvious bug) can go straight to a pull request.

## Building

You need Rust (stable, edition 2024) on Windows 10/11. For the GPU path, an NVIDIA GPU with compute
capability 6.1+ and NVRTC from a CUDA toolkit; see [GPU requirements](README.md#gpu-requirements).

```bash
cargo build --release
target\release\ghost-infer run model.gguf -p "Hello" --gpu
```

`.cargo/config.toml` builds with `target-cpu=native`. That is fine for development, but don't ship
binaries built that way.

## What a good pull request looks like

- **One change per pull request**, with a description of what it does and why.
- **Keep dependencies at two.** ghost-infer depends on `memmap2` and `rayon` and nothing else; a new
  crate needs a strong reason.
- **Style**: the code is not run through `rustfmt`. Match the style around your change, and don't
  reformat code you aren't changing.
- **No regressions in output.** If you touch anything that computes logits (kernels, quantisation,
  attention, the tokenizer, chat templates), run the checks below and say in the pull request what you ran
  and on which GPU.
- **Speed claims need numbers.** For a performance change, give before and after from `ghost-infer bench`
  (same model, same GPU, best of 3), and say which GPU you measured on.
- **Docs**: if you change a command, option, endpoint or environment variable, update the README too.

## Checks

| | |
|---|---|
| `scripts\golden.ps1` | log-probabilities and a greedy reply against the saved reference (`-Save` writes a new one; only do that on purpose) |
| `scripts\accuracy.ps1 [-Gpu]` | fast path vs exact f32 maths at several prompt lengths |
| `scripts\decode-accuracy.ps1` | GPU generation path vs exact maths at up to ~1,500 tokens |
| `scripts\check_dequant.py` | dequantisation vs gguf-py's reference, for every weight format |
| `scripts\kld.py` | full-vocabulary KL divergence against exact maths |

The environment variables in the README's [Accuracy](README.md#accuracy) section (`GHOST_INFER_EXACT`,
`GHOST_INFER_STEPWISE`, `GHOST_INFER_PROFILE` and others) help narrow down where a difference comes from.

## Licence

ghost-infer is dual licensed under MIT and Apache-2.0. Unless you explicitly state otherwise, any
contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or conditions.

## Conduct

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
