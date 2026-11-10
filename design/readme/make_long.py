"""Draws long.svg: long chats stay fast.
Act 1: the context fills to 16K, then 64K, while two bars show writing speed before and after attention moved to
the tensor cores (0.2.4). Act 2: reading a 128K-token prompt, replayed so the old way's ~28 minutes take 6 seconds.
Numbers: Qwen3.6-35B-A3B i1-Q3_K_M on an RTX 2060 SUPER, bench-offclaude-20261110-120233 (Claude closed):
`ghost-infer bench --no-lookup`, now against GHOST_INFER_OLD_WIDE_ATT=1 GHOST_INFER_NO_TC_ATT=1.
Run: python make_long.py   (writes long.svg next to it)"""
import os

T = 16.0
W, H = 860, 404
GREEN = "#7ee2b8"
BAR_X, BAR_W = 236, 500

# measured: generation tok/s after a prompt of that length, before and with the tensor-core kernels
AT_16K = (27.7, 47.3)
AT_64K = (12.3, 49.1)
READ_128K = (77, 507)  # prompt reading, tok/s: about 28 min and 4 min for 131,072 tokens
SCALE = BAR_W / 52.0   # px per tok/s in act 1


def anim(attr, pairs, extra=""):
    k = ";".join(f"{p[0] / T:.4f}" for p in pairs)
    v = ";".join(str(p[1]) for p in pairs)
    return f'<animate attributeName="{attr}" values="{v}" keyTimes="{k}" dur="{T}s" repeatCount="indefinite"{extra}/>'


def show(t0, t1=T - 0.4):
    """opacity: hidden, fades in at t0, out at t1, hidden again before the loop restarts"""
    return anim("opacity", [(0, 0), (t0, 0), (t0 + 0.2, 1), (t1, 1), (min(t1 + 0.3, T), 0), (T, 0)])


out = []
w = out.append
w(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Long chats stay fast: on Qwen3.6-35B-A3B, writing at a 64K context went from 12.0 to 40.8 tokens per second, and reading a 128K-token prompt from 27 minutes to about 4">
  <style>
    .t  {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .h  {{ font-size: 18px; font-weight: 700; fill: #e6edf3; }}
    .s  {{ font-size: 12px; fill: #8b949e; }}
    .k  {{ font-size: 11px; fill: #8b949e; }}
    .sec {{ font-size: 13px; font-weight: 600; fill: #c9d1d9; }}
    .n  {{ font-size: 13px; fill: #8b949e; }}
    .ng {{ fill: {GREEN}; font-weight: 600; }}
    .v  {{ font-size: 13px; font-weight: 700; fill: #8b949e; }}
    .vg {{ fill: {GREEN}; }}
    .b  {{ font-size: 11px; font-weight: 700; }}
  </style>
  <defs>
    <linearGradient id="lg" x1="0" x2="1"><stop offset="0" stop-color="#3fb68b"/><stop offset="1" stop-color="{GREEN}"/></linearGradient>
  </defs>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="#0d1117" stroke="#30363d"/>
  <text class="t h" x="24" y="40">Long chats stay fast</text>
  <text class="t s" x="24" y="62">Attention runs on the tensor cores, so writing speed holds as the chat grows, and long prompts read {READ_128K[1] / READ_128K[0]:.1f}× faster.</text>''')

# ---- act 1: the context fills; writing speed before and now
F16, F64 = (0.6, 2.4), (4.6, 6.6)  # when the context fills to 16K, then to 64K
w('  <text class="t sec" x="24" y="100">Writing, as the chat grows</text>')
w('  <rect x="24" y="112" width="812" height="10" rx="5" fill="#21262d"/>')
w(f'  <rect x="24" y="112" width="0" height="10" rx="5" fill="#30363d">'
  + anim("width", [(0, 0), (F16[0], 0), (F16[1], 203), (F64[0], 203), (F64[1], 812), (T - 0.4, 812), (T - 0.1, 0), (T, 0)],
         ' calcMode="spline" keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1;0.4 0 0.2 1;0 0 1 1;0 0 1 1;0 0 1 1"') + '</rect>')
# the ruler: 0, 16K, 32K, 48K, 64K
for i, lab in enumerate(["0", "16K", "32K", "48K", "64K tokens of context"]):
    x = 24 + i * 203
    anchor = "start" if i == 0 else ("end" if i == 4 else "middle")
    w(f'  <line x1="{x}" y1="124" x2="{x}" y2="129" stroke="#484f58"/>')
    w(f'  <text class="t k" x="{x}" y="142" text-anchor="{anchor}">{lab}</text>')

lanes = [("before 0.2.4", 0, "#6e7681", 174), ("tensor cores", 1, "url(#lg)", 206)]
for name, i, fill, y in lanes:
    a, b = AT_16K[i] * SCALE, AT_64K[i] * SCALE
    w(f'  <text class="t n {"ng" if i else ""}" x="24" y="{y + 4}">{name}</text>')
    w(f'  <rect x="{BAR_X}" y="{y - 6}" width="{BAR_W}" height="12" rx="6" fill="#161b22"/>')
    w(f'  <rect x="{BAR_X}" y="{y - 6}" width="0" height="12" rx="6" fill="{fill}">'
      + anim("width", [(0, 0), (F16[0], 0), (F16[1], f"{a:.1f}"), (F64[0], f"{a:.1f}"), (F64[1], f"{b:.1f}"),
                       (T - 0.4, f"{b:.1f}"), (T - 0.1, 0), (T, 0)],
             ' calcMode="spline" keySplines="0 0 1 1;0.4 0 0.2 1;0 0 1 1;0.4 0 0.2 1;0 0 1 1;0 0 1 1;0 0 1 1"') + '</rect>')
    # the reading beside each bar's end: one at 16K, one at 64K
    for (t0, t1), v, x in (((F16[1], F64[0]), AT_16K[i], BAR_X + a), ((F64[1], T - 0.4), AT_64K[i], BAR_X + b)):
        w(f'  <text class="t v {"vg" if i else ""}" x="{x + 8:.0f}" y="{y + 4}" opacity="0">{v:.1f} tok/s{show(t0, t1)}</text>')

ratio = AT_64K[1] / AT_64K[0]
w(f'''  <g opacity="0">{show(F64[1] + 0.3)}
    <rect x="712" y="86" width="124" height="20" rx="10" fill="{GREEN}"/>
    <text class="t b" x="774" y="100" text-anchor="middle" fill="#0d1117">{ratio:.1f}× faster at 64K</text>
  </g>''')

# ---- act 2: reading a 128K-token prompt; 27 minutes play in 6 seconds
R0, RLEN = 7.6, 6.0
old_min, new_min = 131072 / READ_128K[0] / 60, 131072 / READ_128K[1] / 60
t_new = R0 + RLEN * new_min / old_min
w('  <line x1="24" y1="244" x2="836" y2="244" stroke="#21262d"/>')
w(f'  <text class="t sec" x="24" y="276">Reading a 128K-token prompt</text>')
w(f'  <text class="t k" x="836" y="276" text-anchor="end">replayed: {old_min:.0f} minutes in {RLEN:.0f} seconds</text>')
for name, i, fill, y, t_end, mins in (("before 0.2.4", 0, "#6e7681", 314, R0 + RLEN, old_min),
                                      ("tensor cores", 1, "url(#lg)", 346, t_new, new_min)):
    w(f'  <text class="t n {"ng" if i else ""}" x="24" y="{y + 4}">{name}</text>')
    w(f'  <rect x="{BAR_X}" y="{y - 6}" width="{BAR_W}" height="12" rx="6" fill="#161b22"/>')
    w(f'  <rect x="{BAR_X}" y="{y - 6}" width="0" height="12" rx="6" fill="{fill}">'
      + anim("width", [(0, 0), (R0, 0), (t_end, BAR_W), (T - 0.4, BAR_W), (T - 0.1, 0), (T, 0)]) + '</rect>')
    bw = 76 if mins >= 10 else 66
    w(f'''  <g opacity="0">{show(t_end)}
    <rect x="{BAR_X + BAR_W + 10}" y="{y - 9}" width="{bw}" height="18" rx="9" fill="{GREEN if i else "#30363d"}"/>
    <text class="t b" x="{BAR_X + BAR_W + 10 + bw / 2:.0f}" y="{y + 4}" text-anchor="middle" fill="{"#0d1117" if i else "#c9d1d9"}">✓ {mins:.0f} min</text>
  </g>''')
    w(f'  <text class="t k" x="{BAR_X - 10}" y="{y + 4}" text-anchor="end" opacity="0">{READ_128K[i]} tok/s{show(R0)}</text>')

w(f'  <text class="t s" x="24" y="388">Qwen3.6-35B-A3B i1-Q3_K_M on an RTX 2060 SUPER (8 GB), measured 2026-11-10. Above: writing speed after a prompt that long.</text>')
w("</svg>")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "long.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote long.svg")
