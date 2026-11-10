"""Draws experts.svg: how ghost-infer runs a mixture-of-experts model bigger than the GPU.
Act 1 (loading): the model file splits: attention and shared weights to the GPU, every expert to system RAM,
and the GPU's free memory becomes an expert cache filled from RAM.
Act 2 (generating): a token's experts that are in the cache run on the GPU, the others are computed by the
CPU where they lie in RAM, and an expert that keeps being used moves into the cache.
Run: python make_experts.py   (writes experts.svg next to it)"""
import os

T = 15.0                                   # the loop: loading 0-5 s, three tokens 5-14 s, a pause
END = T - 0.35                             # everything fades out here, to start again
W, H = 860, 430

# system RAM: a grid of experts on a memory stick
COLS, ROWS = 16, 8
CW, CH, GX, GY = 18, 12, 23, 18
RX, RY = 42, 182
# the GPU: the trunk's slots, then the expert cache
TX, TY, TW, TH, TG = 578, 186, 28, 22, 31  # 8 trunk slots
SCOLS, SROWS = 8, 3
SW, SH, SGX, SGY = 26, 17, 31, 24
SX, SY = 578, 244
# the model file
FX, FY, FB, FH = 24, 110, 812 / 40, 22     # 40 blocks: 8 layers of (trunk, 4 expert groups)

GREEN, DIM, LINE = "#7ee2b8", "#2a6b55", "#3fb68b"


def anim(attr, pairs, extra=""):
    k = ";".join(f"{p[0] / T:.4f}" for p in pairs)
    v = ";".join(str(p[1]) for p in pairs)
    return f'<animate attributeName="{attr}" values="{v}" keyTimes="{k}" dur="{T}s" repeatCount="indefinite"{extra}/>'


def window(t0, t1, top=1):
    """Opacity: shown from t0 to t1 (until the loop's end when t1 is None)."""
    if t1 is None:
        return [(0, 0), (t0, 0), (t0 + 0.05, top), (END, top), (END + 0.3, 0), (T, 0)]
    return [(0, 0), (t0, 0), (t0 + 0.08, top), (t1 - 0.08, top), (t1, 0), (T, 0)]


def flash(times, dur=1.1):
    pairs = [(0, 0)]
    for t0 in times:
        pairs += [(t0, 0), (t0 + 0.1, 1), (t0 + dur, 1), (t0 + dur + 0.35, 0)]
    pairs.append((T, 0))
    return pairs


SPL = ' calcMode="spline" keySplines="0 0 1 1;0.45 0 0.2 1;0 0 1 1"'


def fly(x0, y0, w0, h0, x1, y1, w1, h1, start, land, color=GREEN):
    return (f'  <rect x="{x0:.1f}" y="{y0:.1f}" width="{w0:.1f}" height="{h0:.1f}" rx="2.5" fill="{color}" opacity="0">'
            + anim("x", [(0, f"{x0:.1f}"), (start, f"{x0:.1f}"), (land, f"{x1:.1f}"), (T, f"{x1:.1f}")], SPL)
            + anim("y", [(0, f"{y0:.1f}"), (start, f"{y0:.1f}"), (land, f"{y1:.1f}"), (T, f"{y1:.1f}")], SPL)
            + anim("width", [(0, f"{w0:.1f}"), (start, f"{w0:.1f}"), (land, f"{w1:.1f}"), (T, f"{w1:.1f}")], SPL)
            + anim("height", [(0, f"{h0:.1f}"), (start, f"{h0:.1f}"), (land, f"{h1:.1f}"), (T, f"{h1:.1f}")], SPL)
            + anim("opacity", [(0, 0), (start, 0), (start + 0.05, 0.95), (land, 0.95), (land + 0.2, 0), (T, 0)])
            + "</rect>")


def ram(i):
    return RX + (i % COLS) * GX, RY + (i // COLS) * GY


def cache(j):
    return SX + (j % SCOLS) * SGX, SY + (j // SCOLS) * SGY


out = []
w = out.append
w(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="How ghost-infer runs a mixture-of-experts model bigger than the GPU: attention and shared weights go to the GPU, every expert stays in system RAM, the GPU's free memory caches the experts used most; cached experts run on the GPU and the rest on the CPU">
  <style>
    .t  {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .h  {{ font-size: 18px; font-weight: 700; fill: #e6edf3; }}
    .s  {{ font-size: 12px; fill: #8b949e; }}
    .p  {{ font-size: 13px; font-weight: 700; fill: #c9d1d9; }}
    .k  {{ font-size: 11px; fill: #8b949e; }}
    .kg {{ fill: {GREEN}; font-weight: 600; }}
    .kw {{ fill: #c9d1d9; }}
    .cap {{ font-size: 13px; fill: #c9d1d9; }}
    .ln {{ fill: none; stroke: #6e7681; stroke-width: 1.4; }}
    .empty {{ fill: none; stroke: #30363d; stroke-width: 1; stroke-dasharray: 3 2; }}
  </style>
  <defs><filter id="glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="2.5"/></filter></defs>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="#0d1117" stroke="#30363d"/>
  <text class="t h" x="24" y="40">Models bigger than your GPU</text>
  <text class="t s" x="24" y="62">The model is split: what every token needs goes to the GPU, the experts stay in system RAM, and the GPU's free</text>
  <text class="t s" x="24" y="80">memory caches the experts used most. Cached experts run on the GPU, the rest on the CPU. An 80B model runs on 8 GB.</text>''')

# ---- the panels
w(f'''  <text class="t p" x="28" y="152">System RAM</text><text class="t k" x="112" y="152">all the experts</text>
  <path d="M24,160 H420 V350 H258 V357 H246 V350 H24 Z" class="ln"/>
  {"".join(f'<line x1="{x}" y1="351" x2="{x}" y2="357" stroke="#484f58" stroke-width="1.2"/>' for x in list(range(34, 242, 6)) + list(range(264, 416, 6)))}
  <text class="t p" x="566" y="152" fill="{GREEN}" style="fill:{GREEN}">GPU</text><text class="t k" x="598" y="152">8 GB of VRAM</text>
  <rect x="562" y="160" width="274" height="190" rx="9" fill="#161b22" stroke="#8b949e" stroke-width="1.6"/>
  <text class="t k" x="578" y="181">attention + shared weights</text>
  <text class="t k" x="578" y="239">expert cache</text>''')

# the CPU chip on the RAM panel, lit while it computes experts
w(f'''  <g transform="translate(372 140)">
    <rect x="0" y="0" width="40" height="18" rx="4" fill="#161b22" stroke="#6e7681"/>
    {"".join(f'<line x1="{6 + 7 * i}" y1="-3" x2="{6 + 7 * i}" y2="0" stroke="#6e7681"/><line x1="{6 + 7 * i}" y1="18" x2="{6 + 7 * i}" y2="21" stroke="#6e7681"/>' for i in range(5))}
    <text class="t k kw" x="20" y="13" text-anchor="middle">CPU</text>
    <rect x="0" y="0" width="40" height="18" rx="4" fill="#e3b341" opacity="0" filter="url(#glow)">{anim("opacity", flash([5.5, 8.5], 1.1))}</rect>
    <rect x="0" y="0" width="40" height="18" rx="4" fill="none" stroke="#e3b341" stroke-width="1.6" opacity="0">{anim("opacity", flash([5.5, 8.5], 1.1))}</rect>
  </g>''')

# empty places, before loading fills them
for i in range(COLS * ROWS):
    x, y = ram(i)
    w(f'  <rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="2.5" class="empty"/>')
for j in range(8):
    w(f'  <rect x="{TX + j * TG}" y="{TY}" width="{TW}" height="{TH}" rx="3" class="empty"/>')
for j in range(SCOLS * SROWS):
    x, y = cache(j)
    w(f'  <rect x="{x}" y="{y}" width="{SW}" height="{SH}" rx="2.5" class="empty"/>')

# ---- act 1: the model file splits
w(f'  <g>{anim("opacity", [(0, 1), (2.9, 1), (3.2, 0), (END + 0.3, 0), (T, 1)])}')
w(f'    <text class="t k" x="24" y="102">model file</text><text class="t k" x="836" y="102" text-anchor="end">e.g. Qwen3-Coder-Next 80B: 22 GB</text>')
w("  </g>")
ram_land = {}
for b in range(40):
    layer, part = divmod(b, 5)
    fx = FX + b * FB
    if part == 0:  # the trunk: to the GPU
        dx, dy, dw, dh = TX + layer * TG, TY, TW, TH
        start, color = 0.7 + layer * 0.12, LINE
    else:          # four experts' worth: to a run of RAM cells
        g = layer * 4 + part - 1
        dx, dy = ram(g * 4)
        dw, dh = 4 * GX - (GX - CW), CH
        start, color = 1.3 + g * 0.045, "#6e7681"
    land = start + 1.0
    # the block in the file, until it leaves
    w(f'  <rect x="{fx + 0.6:.1f}" y="{FY}" width="{FB - 1.2:.1f}" height="{FH}" rx="2.5" fill="{color}">'
      + anim("opacity", [(0, 1), (start, 1), (start + 0.05, 0), (END + 0.3, 0), (T, 1)]) + "</rect>")
    w(fly(fx + 0.6, FY, FB - 1.2, FH, dx, dy, dw, dh, start, land, color))
    if part == 0:
        w(f'  <rect x="{dx}" y="{dy}" width="{TW}" height="{TH}" rx="3" fill="#1f3b33" stroke="{LINE}" opacity="0">{anim("opacity", window(land, None))}</rect>')
    else:
        for c in range(4):
            ram_land[g * 4 + c] = land
for i in range(COLS * ROWS):
    x, y = ram(i)
    w(f'  <rect x="{x}" y="{y}" width="{CW}" height="{CH}" rx="2.5" fill="#21262d" stroke="#30363d" opacity="0">{anim("opacity", window(ram_land[i], None))}</rect>')

# the cache fills from RAM: 24 experts, spread over the grid
cached = [3, 9, 14, 20, 27, 33, 38, 46, 51, 55, 62, 67, 70, 77, 83, 88, 95, 99, 104, 110, 115, 119, 124, 127]
HOT, OUT = 90, 23            # act 2: expert 90 keeps being used and takes slot 23 (expert 127 leaves)
for j, i in enumerate(cached):
    rx, ry = ram(i)
    sx, sy = cache(j)
    start = 3.0 + j * 0.06
    land = start + 0.9
    w(fly(rx, ry, CW, CH, sx, sy, SW, SH, start, land))
    gone = 11.0 if j == OUT else None
    w(f'  <rect x="{sx}" y="{sy}" width="{SW}" height="{SH}" rx="2.5" fill="{DIM}" stroke="{LINE}" opacity="0">{anim("opacity", window(land, gone) if gone else window(land, None))}</rect>')
    # the RAM copy stays, marked as also cached
    w(f'  <rect x="{rx}" y="{ry}" width="{CW}" height="{CH}" rx="2.5" fill="{DIM}" opacity="0">{anim("opacity", window(land, gone, 0.55) if gone else window(land, None, 0.55))}</rect>')

# ---- act 2: three tokens
# token 1: slots 0 and 5 on the GPU; experts 40 and 90 on the CPU
# token 2: slots 3, 5, 9 on the GPU; expert 90 on the CPU again
# token 3: expert 90 moves into slot 23; slots 0, 3, 9 and 23 all on the GPU
gpu_hits = {0: [5.5, 12.1], 5: [5.5, 8.5], 3: [8.5, 12.1], 9: [8.5, 12.1], OUT: [12.1]}
cpu_hits = {40: [5.5], HOT: [5.5, 8.5]}
for j, times in gpu_hits.items():
    sx, sy = cache(j)
    w(f'  <rect x="{sx - 2}" y="{sy - 2}" width="{SW + 4}" height="{SH + 4}" rx="4" fill="{GREEN}" filter="url(#glow)" opacity="0">{anim("opacity", flash(times))}</rect>')
    w(f'  <rect x="{sx}" y="{sy}" width="{SW}" height="{SH}" rx="2.5" fill="{GREEN}" opacity="0">{anim("opacity", flash(times))}</rect>')
for i, times in cpu_hits.items():
    rx, ry = ram(i)
    w(f'  <rect x="{rx - 2}" y="{ry - 2}" width="{CW + 4}" height="{CH + 4}" rx="4" fill="#e3b341" filter="url(#glow)" opacity="0">{anim("opacity", flash(times))}</rect>')
    w(f'  <rect x="{rx}" y="{ry}" width="{CW}" height="{CH}" rx="2.5" fill="#e3b341" opacity="0">{anim("opacity", flash(times))}</rect>')
# the hot expert moves in, taking the place of one not used lately
rx, ry = ram(HOT)
sx, sy = cache(OUT)
w(fly(rx, ry, CW, CH, sx, sy, SW, SH, 11.1, 12.0))
w(f'  <rect x="{sx}" y="{sy}" width="{SW}" height="{SH}" rx="2.5" fill="{DIM}" stroke="{LINE}" opacity="0">{anim("opacity", window(12.0, None))}</rect>')
w(f'  <rect x="{rx}" y="{ry}" width="{CW}" height="{CH}" rx="2.5" fill="{DIM}" opacity="0">{anim("opacity", window(12.0, None, 0.55))}</rect>')

# the router between the panels
pulse = [(0, "#6e7681")]
for t0 in (5.0, 8.0, 11.0):
    pulse += [(t0 + 0.05, GREEN), (t0 + 0.9, "#6e7681")]
pulse.append((T, "#6e7681"))
w(f'''  <path d="M420,255 H466" class="ln" stroke-dasharray="4 3"/><path d="M516,255 H562" class="ln" stroke-dasharray="4 3"/>
  <circle cx="491" cy="255" r="25" fill="#161b22" stroke="#6e7681" stroke-width="1.4">{anim("stroke", pulse)}</circle>
  <text class="t k kw" x="491" y="251" text-anchor="middle">router</text>''')
for n, t0 in enumerate((5.0, 8.0, 11.0)):
    w(f'  <text class="t k kg" x="491" y="266" text-anchor="middle" opacity="0">token {n + 1}{anim("opacity", window(t0, t0 + 3 if n < 2 else None))}</text>')
w(f'  <text class="t k" x="491" y="266" text-anchor="middle">loading{anim("opacity", [(0, 1), (4.9, 1), (5.0, 0), (END + 0.3, 0), (T, 1)])}</text>')

# ---- captions
caps = [
    (0.0, 3.0, "Loading: attention and shared weights go to the GPU, every expert to system RAM", ""),
    (3.0, 5.0, "The GPU's free memory becomes an expert cache, filled from RAM", ""),
    (5.0, 8.0, "Token 1: 2 of its experts are cached and run on the GPU; 2 run on the CPU, in RAM", "2 GPU · 2 CPU"),
    (8.0, 11.0, "Token 2: 3 on the GPU, 1 on the CPU. That one keeps being used…", "3 GPU · 1 CPU"),
    (11.0, None, "Token 3: …so it moved into the cache. Now all 4 run on the GPU", "4 GPU · 0 CPU"),
]
for t0, t1, text, tally in caps:
    w(f'  <g opacity="0">{anim("opacity", window(t0, t1))}')
    w(f'    <text class="t cap" x="24" y="384">{text}</text>')
    if tally:
        w(f'    <text class="t k kg" x="836" y="384" text-anchor="end">{tally}</text>')
    w("  </g>")
w('  <text class="t s" x="24" y="404">Drawn small: real models have far more experts (Qwen3-Coder-Next has 512 a layer and uses 10 per token).</text>')
w('  <text class="t s" x="24" y="420">When even system RAM is too small, the experts that do not fit are read from the SSD as they are needed.</text>')
w("</svg>")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "experts.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote experts.svg")
