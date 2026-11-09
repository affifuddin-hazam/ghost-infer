"""A PEFT LoRA adapter (adapter_config.json + adapter_model.safetensors) as a GGUF adapter for ghost-infer's
--lora, in llama.cpp's format: general.type "adapter", adapter.type "lora", adapter.lora.alpha, and per
adapted weight blk.N.<name>.weight.lora_a (numpy shape (r, in), so ne = [in, r]) and .lora_b ((out, r)).

  python -I scripts/lora-to-gguf.py <peft dir> <out.gguf> [--arch qwen3] [--type f32|f16]

Llama-style names only (q/k/v/o_proj, gate/up/down_proj), which covers Qwen2/Qwen3 and Llama. rsLoRA's
alpha / sqrt(r) is folded into the alpha written, since the engine scales by alpha / r.
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np
import gguf
from safetensors.numpy import load_file

NAMES = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
KEY = re.compile(r"^(?:base_model\.model\.)?model\.layers\.(\d+)\.(.+)\.lora_([AB])(?:\.default)?\.weight$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("peft")
    ap.add_argument("out")
    ap.add_argument("--arch", default="qwen3")
    ap.add_argument("--type", default="f32", choices=["f32", "f16"])
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.peft, "adapter_config.json"), encoding="utf-8"))
    r = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    if cfg.get("use_rslora"):
        alpha = alpha * math.sqrt(r)  # scale alpha / sqrt(r) == alpha' / r
    if cfg.get("use_dora"):
        sys.exit("DoRA adapters are not supported")
    if cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        sys.exit("per-layer rank_pattern / alpha_pattern is not supported")

    weights = load_file(os.path.join(args.peft, "adapter_model.safetensors"))
    pairs = {}
    for k, v in weights.items():
        m = KEY.match(k)
        if not m:
            sys.exit(f"unexpected tensor {k} (only LoRA A/B on llama-style projections are supported)")
        layer, mod, ab = int(m.group(1)), m.group(2), m.group(3)
        if mod not in NAMES:
            sys.exit(f"{k}: {mod} is not a projection the engine adapts")
        pairs.setdefault((layer, NAMES[mod]), {})[ab] = v

    dtype = np.float32 if args.type == "f32" else np.float16
    w = gguf.GGUFWriter(args.out, args.arch)
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)
    for (layer, name), ab in sorted(pairs.items()):
        if set(ab) != {"A", "B"}:
            sys.exit(f"layer {layer} {name}: missing lora_{'B' if 'A' in ab else 'A'}")
        a, b = ab["A"].astype(np.float32), ab["B"].astype(np.float32)
        if a.shape[0] != r or b.shape[1] != r:
            sys.exit(f"layer {layer} {name}: A {a.shape}, B {b.shape} do not have rank {r}")
        base = f"blk.{layer}.{name}.weight"
        w.add_tensor(base + ".lora_a", np.ascontiguousarray(a.astype(dtype)))
        w.add_tensor(base + ".lora_b", np.ascontiguousarray(b.astype(dtype)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{args.out}: {len(pairs)} pairs, rank {r}, alpha {alpha}")


if __name__ == "__main__":
    main()
