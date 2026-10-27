"""Next-token accuracy of ghost-infer modes against exact CPU maths: full-vocabulary KL divergence
and top-1 agreement over a set of prompts.

usage: python -I scripts\\kld.py <exe> <model.gguf> <mode> [<mode> ...]   (needs numpy)
modes: gpu-f16, gpu-int8 (GPU with that KV cache), cpu (fast CPU path), gpu-step / gpu-step-nocache (the
prompt fed token by token: the generation path, with or without the GPU expert cache). The reference is
exact CPU maths (GHOST_INFER_EXACT), computed once per model and prompt and kept in the temp folder.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

exe, model, modes = sys.argv[1], sys.argv[2], sys.argv[3:]
base = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg developed "
        "movable metal type in Mainz, and printing soon spread across Europe. ")
prompts = [
    "The capital of France is",
    'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n',
    "Water boils at a temperature of",
    "<|im_start|>user\nName three primary colours.<|im_end|>\n<|im_start|>assistant\n",
    "In 1969, the first humans to walk on the Moon were",
    base * 3 + "The printing press was invented by",
    base * 12 + "Movable type was developed in the city of",
    base * 40 + "This invention made books far",
]
ENV = {"exact": {"GHOST_INFER_EXACT": "1"}, "cpu": {}, "gpu-f16": {"GHOST_INFER_KV": "f16"}, "gpu-int8": {"GHOST_INFER_KV": "int8"},
       # the prompt fed one token at a time: the decode path, with and without GPU-resident experts
       "gpu-step": {"GHOST_INFER_STEPWISE": "1"}, "gpu-step-nocache": {"GHOST_INFER_STEPWISE": "1", "GHOST_INFER_EXPERT_CACHE_MB": "0"}}


def logprobs(prompt, mode):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "logits.bin")
        env = dict(os.environ, GHOST_INFER_LOGITS_OUT=out, **ENV[mode])
        args = [exe, "top", model, prompt] + (["--gpu"] if mode.startswith("gpu") else [])
        subprocess.run(args, env=env, check=True, capture_output=True)
        z = np.fromfile(out, dtype=np.float32).astype(np.float64)
    z -= z.max()
    return z - np.log(np.exp(z).sum())


def exact(i, p):
    # the slow reference, kept in the temp folder per model and prompt
    import hashlib
    key = hashlib.sha1((os.path.basename(model) + "\0" + p).encode()).hexdigest()[:16]
    path = os.path.join(tempfile.gettempdir(), f"ghost_infer_exact_{key}.npy")
    if os.path.exists(path):
        return np.load(path)
    ref = logprobs(p, "exact")
    np.save(path, ref)
    return ref


stats = {m: [] for m in modes}
for i, p in enumerate(prompts, 1):
    ref = exact(i, p)
    pe = np.exp(ref)
    line = f"prompt {i} ({len(p):>5} chars)"
    for m in modes:
        lp = logprobs(p, m)
        kl = float((pe * (ref - lp)).sum())
        same = int(ref.argmax() == lp.argmax())
        stats[m].append((kl, same))
        line += f"  {m}: KL {kl:.5f} top-1 {'same' if same else 'DIFF'}"
    print(line, flush=True)
for m in modes:
    kls = [k for k, _ in stats[m]]
    print(f"{m:9} mean KL {np.mean(kls):.5f}  worst KL {max(kls):.5f}  top-1 agrees {sum(s for _, s in stats[m])}/{len(prompts)}")
