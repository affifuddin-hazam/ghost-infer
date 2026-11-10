"""Draws small.svg: ghost-infer is one small program.
A typical local-AI stack piles up, then folds into one 3.8 MB file, which starts a server in a second.
Run: python make_small.py   (writes small.svg next to it)"""
import os

T = 12.0
END = T - 0.4
W, H = 860, 400
GREEN, LINE = "#7ee2b8", "#3fb68b"


def anim(attr, pairs, extra=""):
    k = ";".join(f"{p[0] / T:.4f}" for p in pairs)
    v = ";".join(str(p[1]) for p in pairs)
    return f'<animate attributeName="{attr}" values="{v}" keyTimes="{k}" dur="{T}s" repeatCount="indefinite"{extra}/>'


def window(t0, t1=None, top=1):
    if t1 is None:
        return [(0, 0), (t0, 0), (t0 + 0.12, top), (END, top), (END + 0.3, 0), (T, 0)]
    return [(0, 0), (t0, 0), (t0 + 0.12, top), (t1 - 0.12, top), (t1, 0), (T, 0)]


SPL = ' calcMode="spline" keySplines="0 0 1 1;0.5 0 0.3 1;0 0 1 1"'
out = []
w = out.append
w(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="ghost-infer is one small program: instead of Python, pip packages, a CUDA toolkit and Docker, one 3.8 MB file with two dependencies that starts a server">
  <style>
    .t  {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .m  {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; }}
    .h  {{ font-size: 18px; font-weight: 700; fill: #e6edf3; }}
    .s  {{ font-size: 12px; fill: #8b949e; }}
    .k  {{ font-size: 11px; fill: #8b949e; }}
    .box {{ font-size: 13px; fill: #c9d1d9; }}
    .big {{ font-size: 44px; font-weight: 800; fill: {GREEN}; letter-spacing: -1px; }}
    .term {{ font-size: 12.5px; fill: #c9d1d9; }}
    .dim {{ fill: #8b949e; }}
    .gr {{ fill: {GREEN}; }}
  </style>
  <defs>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6"/></filter>
  </defs>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="#0d1117" stroke="#30363d"/>
  <text class="t h" x="24" y="40">One small program</text>
  <text class="t s" x="24" y="62">No Python, no Docker, no CUDA toolkit to install: one file that compiles its GPU code for your card on first start.</text>''')

# ---- the usual pile
w(f'  <text class="t k" x="24" y="98">what a local AI setup often asks for{anim("opacity", [(0, 1), (3.4, 1), (4.0, 0), (END + 0.3, 0), (T, 1)])}</text>')
w(f'  <text class="t box" x="354" y="166" text-anchor="end" opacity="0" style="fill:#8b949e">all of that,{anim("opacity", window(4.4))}</text>')
w(f'  <text class="t box" x="354" y="186" text-anchor="end" opacity="0" style="fill:#8b949e">in one file{anim("opacity", window(4.6))}</text>')
pile = ["Python and a virtual environment", "pip install torch (the CUDA build)", "the CUDA toolkit", "Docker Desktop", "a web UI, and its own server"]
for n, label in enumerate(pile):
    y = 108 + n * 34
    appear = 0.3 + n * 0.45
    fold = 3.4 + (len(pile) - 1 - n) * 0.08
    w(f'  <g opacity="0">')
    w("    " + anim("opacity", [(0, 0), (appear, 0), (appear + 0.25, 1), (fold, 1), (fold + 0.7, 0), (T, 0)]))
    w("    " + f'<animateTransform attributeName="transform" type="translate" values="0 -14;0 -14;0 0;0 0;330 0;330 0" keyTimes="0;{appear / T:.4f};{(appear + 0.3) / T:.4f};{fold / T:.4f};{(fold + 0.7) / T:.4f};1" dur="{T}s" repeatCount="indefinite" calcMode="spline" keySplines="0 0 1 1;0.3 0 0.2 1;0 0 1 1;0.6 0 0.9 0.6;0 0 1 1"/>')
    w(f'    <rect x="24" y="{y}" width="330" height="28" rx="6" fill="#161b22" stroke="#30363d"/>')
    w(f'    <circle cx="40" cy="{y + 14}" r="3.5" fill="#6e7681"/>')
    w(f'    <text class="t box" x="52" y="{y + 19}">{label}</text>')
    w("  </g>")

# ---- the arrow
w(f'  <path d="M374,190 H448 M440,183 L448,190 L440,197" fill="none" stroke="#6e7681" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" opacity="0">{anim("opacity", window(3.2))}</path>')

# ---- the one file
w(f'''  <g opacity="0">{anim("opacity", window(3.9))}
    <rect x="468" y="104" width="104" height="132" rx="10" fill="{GREEN}" opacity="0" filter="url(#glow)">{anim("opacity", [(0, 0), (3.9, 0), (4.1, 0.55), (4.9, 0.0), (T, 0)])}</rect>
    <path d="M478,104 H542 L572,134 V226 Q572,236 562,236 H478 Q468,236 468,226 V114 Q468,104 478,104 Z" fill="#161b22" stroke="{GREEN}" stroke-width="1.8"/>
    <path d="M542,104 V126 Q542,134 550,134 H572" fill="none" stroke="{GREEN}" stroke-width="1.8"/>
    <g transform="translate(520 186)">
      <path d="M-22,10 V-12 A22,22 0 0 1 22,-12 V10 L15,4 L8,10 L0,4 L-8,10 L-15,4 Z" fill="none" stroke="{GREEN}" stroke-width="1.8" stroke-linejoin="round"/>
      <circle cx="-8" cy="-10" r="3" fill="{GREEN}"/><circle cx="8" cy="-10" r="3" fill="{GREEN}"/>
    </g>
    <text class="t k" x="520" y="256" text-anchor="middle" style="fill:#c9d1d9">ghost-infer.exe</text>
  </g>
  <g opacity="0">{anim("opacity", window(4.3))}
    <text class="t big" x="604" y="160">3.8 MB</text>
    <text class="t box" x="606" y="188">one file, written in Rust</text>
    <text class="t box" x="606" y="210">2 dependencies: <tspan class="gr">memmap2</tspan> and <tspan class="gr">rayon</tspan></text>
    <text class="t box" x="606" y="232">nothing else to install</text>
  </g>''')

# ---- the terminal: it starts in a second
w('  <rect x="24" y="284" width="812" height="100" rx="8" fill="#010409" stroke="#30363d"/>')
w('  <circle cx="40" cy="298" r="4" fill="#30363d"/><circle cx="54" cy="298" r="4" fill="#30363d"/><circle cx="68" cy="298" r="4" fill="#30363d"/>')
lines = [
    (5.0, 1.4, '<tspan class="gr">&gt;</tspan> ghost-infer serve Qwen3-4B-Instruct-2507-Q4_K_M.gguf --gpu'),
    (6.9, 0.5, '<tspan class="dim">[loaded Qwen3-4B-Instruct-2507 on GPU NVIDIA GeForce RTX 2060 SUPER in 1.1s]</tspan>'),
    (7.5, 0.4, 'chat in your browser: <tspan class="gr">http://127.0.0.1:8080/</tspan>'),
    (8.0, 0.4, 'OpenAI-compatible API: <tspan class="gr">http://127.0.0.1:8080/v1</tspan>'),
]
for n, (t0, dur, text) in enumerate(lines):
    y = 322 + n * 18
    w(f'  <clipPath id="tl{n}"><rect x="36" y="{y - 13}" height="18" width="0">{anim("width", [(0, 0), (t0, 0), (t0 + dur, 800), (END, 800), (END + 0.01, 0), (T, 0)])}</rect></clipPath>')
    w(f'  <text class="m term" x="38" y="{y}" clip-path="url(#tl{n})">{text}</text>')
# the cursor, after the last line
w(f'  <rect x="38" y="{322 + 3 * 18 + 6}" width="8" height="3" fill="{GREEN}" opacity="0">{anim("opacity", [(0, 0), (8.4, 0), (8.45, 1), (8.9, 1), (8.95, 0), (9.4, 0), (9.45, 1), (9.9, 1), (9.95, 0), (10.4, 0), (10.45, 1), (10.9, 1), (10.95, 0), (T, 0)])}</rect>')
w("</svg>")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "small.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote small.svg")
