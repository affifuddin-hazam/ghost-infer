"""Converts the song-writing half of YuE2 (m-a-p/YuE2-3B) to a qwen3 GGUF for `ghost-infer yue2`.

YuE2 is a mixture of transformers: every layer has an AR path (writes the song's codec tokens) and a NAR
path (flow matching, turns them into audio latents). The AR path is exactly a Qwen3 layer (QK-norm, half-split
RoPE, SwiGLU, grouped-query attention), so it goes into a plain qwen3 GGUF. The NAR path stays in PyTorch.

Song tokens are only ever MUSIC_END (151852) or one of the 32768 codec tokens after it, so the output layer
keeps just those 32769 rows (logit i is token 151852 + i). The embedding keeps all 184704 rows (the prompt has
text and ABC tokens); ghost-infer keeps it in RAM.

  python scripts/yue2-convert.py MODEL.safetensors OUT.gguf [--type q8_0|f16]
         [--ref REF.json --take TAKE_DIR]

--ref also writes a test for `ghost-infer yue2-check OUT.gguf REF.json`: a take's prompt and first song tokens,
with the logits YuE2 gives for the next token at a few positions (PyTorch, on the GPU in FP16).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import gguf
from safetensors import safe_open

LOGIT_FROM = 151852  # MUSIC_END; the codec tokens follow it
LOGIT_ROWS = 1 + 32768
CHECK_AT = (0, 1, 2, 8, 32, 63)  # song positions whose next-token logits the test compares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="YuE2-3B model.safetensors")
    ap.add_argument("out")
    ap.add_argument("--type", choices=("q8_0", "f16"), default="q8_0")
    ap.add_argument("--ref")
    ap.add_argument("--take", help="a finished YuE2 take folder (prefix.npy, semantic.npy) for --ref")
    args = ap.parse_args()
    src = Path(args.model)
    cfg = json.loads((src.parent / "config.json").read_text(encoding="utf-8"))
    assert cfg["model_type"] == "yue2", cfg["model_type"]
    convert(src, cfg, Path(args.out), args.type)
    if args.ref:
        if not args.take:
            sys.exit("--ref needs --take")
        reference(src.parent, Path(args.take), Path(args.ref))


def convert(src, cfg, out_path, kind):
    qtype = gguf.GGMLQuantizationType.Q8_0 if kind == "q8_0" else gguf.GGMLQuantizationType.F16
    w = gguf.GGUFWriter(str(out_path), "qwen3")
    w.add_name("YuE2-3B song tokens")
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(cfg["num_hidden_layers"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_key_length(cfg["head_dim"])
    w.add_value_length(cfg["head_dim"])
    w.add_rope_freq_base(float(cfg["rope_theta"]))
    w.add_layer_norm_rms_eps(float(cfg["rms_norm_eps"]))
    w.add_vocab_size(LOGIT_ROWS)  # the logits: the output layer's rows, not the embedding's
    w.add_uint32("yue2.logit_from", LOGIT_FROM)
    w.add_uint32("yue2.embedding_rows", cfg["vocab_size"])
    w.add_bool("ghost.embd_on_host", True)

    with safe_open(str(src), "pt") as f:
        def get(name):
            return f.get_tensor(name).float().numpy()

        def matrix(name, data):
            data = np.ascontiguousarray(data, dtype=np.float32)
            if qtype == gguf.GGMLQuantizationType.Q8_0:
                w.add_tensor(name, gguf.quants.quantize(data, qtype), raw_dtype=qtype)
            else:
                w.add_tensor(name, data.astype(np.float16))

        def vector(name, data):
            w.add_tensor(name, np.ascontiguousarray(data, dtype=np.float32))

        print("embedding", flush=True)
        matrix("token_embd.weight", get("model.embed_tokens.weight"))
        matrix("output.weight", get("lm_head.weight")[LOGIT_FROM:LOGIT_FROM + LOGIT_ROWS])
        vector("output_norm.weight", get("model.norm.weight"))
        for i in range(cfg["num_hidden_layers"]):
            print(f"layer {i}", flush=True)
            p = f"model.layers.{i}."
            b = f"blk.{i}."
            vector(b + "attn_norm.weight", get(p + "input_layernorm.weight"))
            matrix(b + "attn_q.weight", get(p + "self_attn.q_proj.weight"))
            matrix(b + "attn_k.weight", get(p + "self_attn.k_proj.weight"))
            matrix(b + "attn_v.weight", get(p + "self_attn.v_proj.weight"))
            vector(b + "attn_q_norm.weight", get(p + "self_attn.q_norm.weight"))
            vector(b + "attn_k_norm.weight", get(p + "self_attn.k_norm.weight"))
            matrix(b + "attn_output.weight", get(p + "self_attn.o_proj.weight"))
            vector(b + "ffn_norm.weight", get(p + "post_attention_layernorm.weight"))
            matrix(b + "ffn_gate.weight", get(p + "mlp.gate_proj.weight"))
            matrix(b + "ffn_up.weight", get(p + "mlp.up_proj.weight"))
            matrix(b + "ffn_down.weight", get(p + "mlp.down_proj.weight"))
    print("writing", out_path, flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()


def reference(model_dir, take, out):
    """Next-token logits from YuE2 itself (the AR path, FP16 on the GPU) at CHECK_AT song positions."""
    import torch
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.placement import activate_phase

    prefix = np.load(take / "prefix.npy").astype(np.int64).tolist()
    song = np.load(take / "semantic.npy").astype(np.int64).tolist()  # codec values, before the offset
    tokens = [LOGIT_FROM + 1 + s for s in song[:max(CHECK_AT) + 1]]
    model = YuE2ForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True).eval()
    activate_phase(model, "ar", torch.device("cuda"))
    ids = torch.tensor([prefix + tokens], device="cuda")
    with torch.inference_mode():
        logits = model(ids, use_cache=False).logits[0].float()
    checks = []
    for k in CHECK_AT:
        row = logits[len(prefix) - 1 + k, LOGIT_FROM:LOGIT_FROM + LOGIT_ROWS].cpu().numpy()
        checks.append({"after": k, "logits": [round(float(x), 4) for x in row]})
    out.write_text(json.dumps({"prefix": prefix, "song": tokens, "checks": checks}), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
