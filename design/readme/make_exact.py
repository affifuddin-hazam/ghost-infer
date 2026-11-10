"""Draws exact.svg: exact speculative decoding.
Top: the same line of code written two ways. Plain decoding adds one token per step; with guesses, a run of
guessed tokens is checked in one pass, the ones the model agrees with are kept, the first wrong one is replaced by
the model's own token and the rest are dropped. Both end with the same text. (An illustration of the method.)
Bottom: measured gains, bench-offclaude-20261110-120233 (Claude closed). Draft model: JOSIEFIED-Qwen3-8B Q4_K_M
with JOSIEFIED-Qwen3-0.6B Q8_0, the prime-function prompt, 98.6 vs 69.9 tok/s, the same reply. Lookup: its
bench-lookup.ps1 set, Qwen3-4B Q4_K_M, copy-heavy median 312.0 vs 97.3, edit a file 648.6 vs 96.1, 22 of 22
replies the same.
Run: python make_exact.py   (writes exact.svg next to it)"""
import os

T = 13.0
W, H = 860, 392
GREEN = "#7ee2b8"
RED = "#f47067"
CH, PAD, GAP = 7.2, 10, 4  # chip: px per character, inner padding, gap between chips
X0 = 24

TEXT = ["def", " is", "_prime", "(n", "):", " if", " n", " <", " 2", ":", " return", " False"]
# guessed runs: (start time, guesses, how many the model keeps); after a miss the model's own token takes its place
CYCLES = [(0.7, ["def", " is", "_prime", "(n"], 4),
          (2.0, ["):", " return", " n", " >"], 1),
          (3.3, [" n", " <", " 2", ":"], 4),
          (4.6, [" return", " False"], 2)]
PLAIN_T0, PLAIN_STEP = 0.7, 0.66
CHECK, SETTLE, FADE = 0.4, 0.85, 0.45  # sweep starts, results show, the dropped ones are gone (after the cycle start)
END = T - 0.5


def anim(attr, pairs, extra=""):
    k = ";".join(f"{p[0] / T:.4f}" for p in pairs)
    v = ";".join(str(p[1]) for p in pairs)
    return f'<animate attributeName="{attr}" values="{v}" keyTimes="{k}" dur="{T}s" repeatCount="indefinite"{extra}/>'


def window(t0, t1=END, fade=0.2):
    return anim("opacity", [(0, 0), (t0, 0), (t0 + fade, 1), (t1, 1), (min(t1 + 0.3, T), 0), (T, 0)])


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def width(tok):
    return len(tok.strip()) * CH + 2 * PAD


def chip(x, y, tok, kind, t0, t1=END):
    """kind: plain, guess (dashed), kept, own (the model's own token), wrong"""
    style = {
        "plain": ('fill="#161b22" stroke="#30363d"', "#c9d1d9"),
        "guess": ('fill="none" stroke="#6e7681" stroke-dasharray="3 3"', "#8b949e"),
        "kept": (f'fill="#12261f" stroke="{GREEN}"', "#e6edf3"),
        "own": (f'fill="{GREEN}" stroke="{GREEN}"', "#0d1117"),
        "wrong": (f'fill="#2d1517" stroke="{RED}"', RED),
    }[kind]
    w = width(tok)
    t = tok.strip()
    strike = (f'<line x1="{x + 4}" y1="{y + 12}" x2="{x + w - 4:.1f}" y2="{y + 12}" stroke="{RED}" stroke-width="1.4"/>'
              if kind == "wrong" else "")
    return (f'  <g opacity="0">{window(t0, t1, 0.15)}<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" rx="5" {style[0]}/>'
            f'<text class="m" x="{x + PAD:.1f}" y="{y + 16}" textLength="{len(t) * CH:.1f}" lengthAdjust="spacingAndGlyphs" '
            f'fill="{style[1]}">{esc(t)}</text>{strike}</g>')


out = []
w = out.append
w(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Exact speculative decoding: guessed tokens are checked in one pass and only the ones the model agrees with are kept, so the text is the same as plain decoding. A small draft model: 1.4 times faster writing code; lookup: 3.2 times faster on agent tasks that copy text">
  <style>
    .t  {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .m  {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 12px; }}
    .h  {{ font-size: 18px; font-weight: 700; fill: #e6edf3; }}
    .s  {{ font-size: 12px; fill: #8b949e; }}
    .k  {{ font-size: 11px; fill: #8b949e; }}
    .sec {{ font-size: 13px; font-weight: 600; fill: #c9d1d9; }}
    .sg {{ fill: {GREEN}; }}
    .b  {{ font-size: 11px; font-weight: 700; }}
    .big {{ font-size: 34px; font-weight: 800; fill: {GREEN}; letter-spacing: -1px; }}
    .l1 {{ font-size: 13px; font-weight: 600; fill: #e6edf3; }}
    .l2 {{ font-size: 11.5px; fill: #8b949e; }}
  </style>
  <defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.5"/></filter></defs>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="#0d1117" stroke="#30363d"/>
  <text class="t h" x="24" y="40">Exact speculative decoding</text>
  <text class="t s" x="24" y="62">Guess a few tokens, check them all in one pass, keep only the ones the model agrees with: the same text as plain decoding, faster.</text>''')

# ---- plain decoding: one token per step
YA, YB = 112, 186
w(f'  <text class="t sec" x="24" y="{YA - 10}">Plain decoding <tspan class="k" font-weight="400">one token per step</tspan></text>')
x = X0
for i, tok in enumerate(TEXT):
    w(chip(x, YA, tok, "plain", PLAIN_T0 + i * PLAIN_STEP))
    x += width(tok) + GAP
plain_done = PLAIN_T0 + (len(TEXT) - 1) * PLAIN_STEP + 0.2
w(f'''  <g opacity="0">{window(plain_done)}<rect x="752" y="{YA + 2}" width="84" height="20" rx="10" fill="#30363d"/>
    <text class="t b" x="794" y="{YA + 16}" text-anchor="middle" fill="#c9d1d9">✓ {len(TEXT)} steps</text></g>''')

# ---- with guesses: checked a run at a time
w(f'  <text class="t sec sg" x="24" y="{YB - 10}">With guesses <tspan class="k" font-weight="400">a run of guesses checked in one pass</tspan></text>')
x = X0
done = 0
for t0, guesses, keep in CYCLES:
    gx = x
    run_w = sum(width(g) + GAP for g in guesses) - GAP
    for j, g in enumerate(guesses):
        if j < keep:  # kept: dashed while guessed, then solid
            w(chip(gx, YB, g, "guess", t0, t0 + SETTLE))
            w(chip(gx, YB, g, "kept", t0 + SETTLE))
        elif j == keep:  # the first miss: red, then the model's own token in its place
            w(chip(gx, YB, g, "guess", t0, t0 + SETTLE))
            w(chip(gx, YB, g, "wrong", t0 + SETTLE, t0 + SETTLE + FADE))
        else:  # the rest of the run is dropped
            w(chip(gx, YB, g, "guess", t0, t0 + SETTLE + 0.1))
        gx += width(g) + GAP
    # the check: one sweep across the whole run
    w(f'  <rect x="{x - 3:.1f}" y="{YB - 4}" width="2.5" height="32" rx="1" fill="{GREEN}" filter="url(#glow)" opacity="0">'
      + anim("opacity", [(0, 0), (t0 + CHECK, 0), (t0 + CHECK + 0.05, 1), (t0 + SETTLE - 0.05, 1), (t0 + SETTLE, 0), (T, 0)])
      + anim("x", [(0, f"{x - 3:.1f}"), (t0 + CHECK, f"{x - 3:.1f}"), (t0 + SETTLE, f"{x + run_w + 1:.1f}"), (T, f"{x + run_w + 1:.1f}")])
      + "</rect>")
    # advance past what was kept; after a miss, the model's own token goes in at once
    for g in guesses[:keep]:
        x += width(g) + GAP
    done += keep
    if keep < len(guesses):
        own = TEXT[done]
        w(chip(x, YB, own, "own", t0 + SETTLE + FADE))
        x += width(own) + GAP
        done += 1
assert done == len(TEXT), done
spec_done = CYCLES[-1][0] + SETTLE + 0.2
w(f'''  <g opacity="0">{window(spec_done)}<rect x="752" y="{YB + 2}" width="84" height="20" rx="10" fill="{GREEN}"/>
    <text class="t b" x="794" y="{YB + 16}" text-anchor="middle" fill="#0d1117">✓ {len(CYCLES)} checks</text></g>''')

# ---- the same text: a bracket between the rows once both are done
sx = 24 + sum(width(g) + GAP for g in TEXT) + 8
w(f'''  <g opacity="0">{window(plain_done + 0.3)}
    <path d="M{sx:.0f},{YA + 12} h10 V{YB + 12} h-10" fill="none" stroke="{GREEN}" stroke-width="1.6"/>
    <text class="t b" x="{sx + 18:.0f}" y="{(YA + YB) / 2 + 16:.0f}" fill="{GREEN}">same text, byte for byte</text>
  </g>''')
w(f'  <text class="t k" x="836" y="{YB + 48}" text-anchor="end" font-style="italic">an illustration of the method; the numbers below are measured</text>')

# ---- measured
TY = 252
tiles = [
    ("1.4×", "a small model guesses", "Qwen3 8B with Qwen3-0.6B guessing, writing code:",
     "69.9 → 98.6 tok/s"),
    ("3.2×", "lookup: guesses copied from the chat", "Qwen3-4B on six agent tasks that copy text (edit,",
     "rename, tool calls): 97 → 312 tok/s, up to 6.7× on an edit"),
]
for n, (big, l1, a, b) in enumerate(tiles):
    tx = 24 + n * 412
    w(f'''  <g opacity="0">{window(0.2 + n * 0.2, END + 0.3)}
    <rect x="{tx}" y="{TY}" width="400" height="116" rx="10" fill="#161b22" stroke="#30363d"/>
    <text class="t big" x="{tx + 18}" y="{TY + 46}">{big}</text>
    <text class="t l1" x="{tx + 18}" y="{TY + 70}">{l1}</text>
    <text class="t l2" x="{tx + 18}" y="{TY + 89}">{a}</text>
    <text class="t l2" x="{tx + 18}" y="{TY + 105}">{b}</text>
  </g>''')
w(f'  <text class="t k" x="24" y="{H - 10}">Greedy, RTX 2060 SUPER, measured 2026-11-10. In every run, the reply with guessing was the same as without it.</text>')
w("</svg>")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "exact.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote exact.svg")
