"""Compare betterllama's dequantisation with gguf-py's reference for a few rows of every weight format."""
import subprocess
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

exe, path = sys.argv[1], sys.argv[2]
reader = GGUFReader(path)
by_type = {}
for t in reader.tensors:
    name = t.tensor_type.name
    if name in ("F32", "F16"):
        continue
    by_type.setdefault(name, []).append(t)

worst_all = 0.0
for ty, tensors in sorted(by_type.items()):
    # the first, a 3-D expert tensor if there is one, and the last of this format
    picks = {tensors[0].name: tensors[0], tensors[-1].name: tensors[-1]}
    for t in tensors:
        if len(t.shape) == 3:
            picks[t.name] = t
            break
    for t in picks.values():
        rows = t.data.shape[0] if t.data.ndim == 2 else int(np.prod(t.data.shape[:-1]))
        raw = t.data.reshape(rows, -1)
        for r in (0, rows // 2 + 7, rows - 1):
            want = dequantize(raw[r], t.tensor_type).astype(np.float32).ravel()
            out = subprocess.run([exe, "dequant", path, t.name, str(r)], capture_output=True, text=True, check=True).stdout.split()
            got = np.array([float(v) for v in out], dtype=np.float32)
            scale = max(float(np.abs(want).max()), 1e-12)
            err = float(np.abs(got - want).max()) / scale
            worst_all = max(worst_all, err)
            print(f"{ty:5} {t.name:32} row {r:>7}: {len(got)} values, max |diff| / max |value| = {err:.2e}")
print(f"worst relative difference: {worst_all:.2e}")
