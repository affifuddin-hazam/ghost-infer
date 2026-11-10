"""Draws sleep.svg: it learns while you sleep (the nightly LoRA, scripts/nightly).
Day: chats come in and a few things in them are worth keeping. 3:00: those facts are trained into a small
adapter beside the model's own frozen weights, then checked (facts learned, general answers kept, tools still
called) before it is installed. Morning: the model knows them without being told, and still uses its tools.
The facts shown are made-up examples. Numbers: runs/20261110-134255-toolfix (Qwen3-4B, 21 facts: held-out fact
loss 6.36 -> 0.66, tools 6 of 6, ~25 minutes of training on the RTX 2060 SUPER).
Abstract shapes only: no real app's UI.
Run: python make_sleep.py   (writes sleep.svg next to it)"""
import os

T = 17.0
END = T - 0.5
W, H = 860, 486
GREEN = "#7ee2b8"
MOON = "#c9d1d9"

# the clock
DAY_END, NIGHT_END = 4.2, 11.2


def anim(attr, pairs, extra=""):
    k = ";".join(f"{p[0] / T:.4f}" for p in pairs)
    v = ";".join(str(p[1]) for p in pairs)
    return f'<animate attributeName="{attr}" values="{v}" keyTimes="{k}" dur="{T}s" repeatCount="indefinite"{extra}/>'


def show(t0, t1=END, fade=0.25):
    return anim("opacity", [(0, 0), (t0, 0), (t0 + fade, 1), (t1, 1), (min(t1 + 0.3, T), 0), (T, 0)])


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


GHOST = ("M4,11 C4,6.2 7.6,2.5 12,2.5 C16.4,2.5 20,6.2 20,11 L20,21 Q18.67,18.6 17.33,21 Q16,23.4 14.67,21 "
         "Q13.33,18.6 12,21 Q10.67,23.4 9.33,21 Q8,18.6 6.67,21 Q5.33,23.4 4,21 Z")

out = []
w = out.append
w(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="It learns while you sleep: each night ghost-infer trains what the day's chats taught into a small adapter beside the model's weights, checks that it learned the facts, kept its general answers and still calls its tools, and installs it; the next morning the model knows those facts without being told">
  <style>
    .t  {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .m  {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; }}
    .h  {{ font-size: 18px; font-weight: 700; fill: #e6edf3; }}
    .s  {{ font-size: 12px; fill: #8b949e; }}
    .k  {{ font-size: 11px; fill: #8b949e; }}
    .sec {{ font-size: 12px; font-weight: 700; fill: #8b949e; letter-spacing: .06em; }}
    .chip {{ font-size: 11px; fill: #0d1117; font-weight: 600; }}
    .say {{ font-size: 12px; fill: #c9d1d9; }}
    .ok {{ font-size: 12px; fill: #c9d1d9; }}
    .b  {{ font-size: 11px; font-weight: 700; }}
  </style>
  <defs>
    <linearGradient id="lg" x1="0" x2="1"><stop offset="0" stop-color="#3fb68b"/><stop offset="1" stop-color="{GREEN}"/></linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3"/></filter>
  </defs>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="#0d1117" stroke="#30363d"/>
  <text class="t h" x="24" y="40">It learns while you sleep</text>
  <text class="t s" x="24" y="62">At night it trains what your chats taught it into a small adapter beside the model's own weights, and checks it before it is used.</text>''')

# ---- the sky: a strip that goes dark at night, a sun and a moon crossing it
SKY_Y, SKY_H = 80, 34
w(f'  <rect x="24" y="{SKY_Y}" width="812" height="{SKY_H}" rx="8" fill="#161b22">'
  + anim("fill", [(0, "#16202b"), (DAY_END - 0.4, "#16202b"), (DAY_END + 0.4, "#0a0d16"), (NIGHT_END - 0.3, "#0a0d16"),
                  (NIGHT_END + 0.5, "#16202b"), (T, "#16202b")]) + '</rect>')
for i, (x, y) in enumerate([(320, 92), (372, 104), (430, 90), (488, 100), (540, 93), (590, 106)]):  # stars at night
    w(f'  <circle cx="{x}" cy="{y}" r="1.2" fill="{MOON}" opacity="0">{show(DAY_END + 0.3 + i * 0.12, NIGHT_END - 0.4)}</circle>')
for label, x in (("day", 140), ("3:00", 440), ("next morning", 722)):
    w(f'  <text class="t k" x="{x}" y="{SKY_Y + 21}" text-anchor="middle" opacity="0.9">{label}</text>')
# the sun sets, the moon crosses, the sun comes back
w(f'  <circle cy="{SKY_Y + 17}" r="7" fill="#f2cc60">'
  + anim("cx", [(0, 60), (DAY_END, 250), (DAY_END + 0.01, 650), (NIGHT_END, 650), (END, 800), (T, 800)])
  + anim("opacity", [(0, 1), (DAY_END - 0.3, 1), (DAY_END, 0), (NIGHT_END, 0), (NIGHT_END + 0.3, 1), (END, 1), (T, 0.0)]) + '</circle>')
w(f'  <g opacity="0">{show(DAY_END, NIGHT_END - 0.2)}<g>'
  + f'<animateTransform attributeName="transform" type="translate" values="300 0;300 0;580 0;580 0" keyTimes="0;{DAY_END / T:.4f};{NIGHT_END / T:.4f};1" dur="{T}s" repeatCount="indefinite"/>'
  + f'<path d="M0,{SKY_Y + 9} a8,8 0 1 0 8,13 a6.5,6.5 0 1 1 -8,-13 Z" fill="{MOON}"/></g></g>')

# ---- columns
TOP = 136
cols = [(24, 252, "THE DAY'S CHATS"), (296, 268, "AT 3:00, ON THE GPU"), (588, 248, "THE NEXT MORNING")]
for x, wd, name in cols:
    w(f'  <rect x="{x}" y="{TOP}" width="{wd}" height="{H - TOP - 44}" rx="10" fill="#0f141b" stroke="#21262d"/>')
    w(f'  <text class="t sec" x="{x + 14}" y="{TOP + 22}">{name}</text>')

# ---- the day: chat lines, three of them hold something worth keeping
facts = ["deploys on Fridays", "API lives in src/server", "prefers short answers"]
lines = [(0.3, 180, None), (0.7, 130, 0), (1.1, 200, None), (1.5, 150, 1), (1.9, 110, None), (2.3, 170, 2)]
def chip_w(text):
    return text_w(text) + 16  # 8 px either side of the text


def text_w(text):
    return len(text) * 5.3  # forced with textLength, so the margins are even in any font


for n, (t0, width, fact) in enumerate(lines):
    if fact is not None:  # a bubble holding a fact is as wide as the fact needs, with the same margin each side
        width = chip_w(facts[fact]) + 16
    y = TOP + 40 + n * 34
    me = n % 2 == 0
    x = 38 if me else 38 + 26
    w(f'  <g opacity="0">{show(t0, NIGHT_END + 0.4)}')
    w(f'    <rect x="{x}" y="{y}" width="{width:.0f}" height="22" rx="11" fill="{"#1c2430" if me else "#161b22"}" stroke="#30363d"/>')
    if fact is not None:
        fx = x + 8
        fw = chip_w(facts[fact])
        w(f'    <rect x="{fx}" y="{y + 3}" width="{fw:.0f}" height="16" rx="8" fill="{GREEN}" opacity="0">{show(2.8 + fact * 0.15, NIGHT_END + 0.4)}</rect>')
        w(f'    <text class="t chip" x="{fx + 8}" y="{y + 15}" textLength="{text_w(facts[fact]):.1f}" lengthAdjust="spacingAndGlyphs" opacity="0">{esc(facts[fact])}{show(2.8 + fact * 0.15, NIGHT_END + 0.4)}</text>')
    else:
        for j in range(3):  # unread words: short bars
            w(f'    <rect x="{x + 12 + j * (width - 24) / 3:.0f}" y="{y + 9}" width="{(width - 24) / 3 - 8:.0f}" height="4" rx="2" fill="#30363d"/>')
    w('  </g>')
w(f'  <text class="t k" x="38" y="{H - 58}" opacity="0">a model reads them for durable facts{show(2.8, NIGHT_END + 0.4)}</text>')

# ---- the night: the model's layers (frozen) and the adapter growing beside each
LX, LY, LW, LH, GAP = 318, TOP + 48, 140, 16, 6
TRAIN0, TRAIN1 = DAY_END + 1.4, DAY_END + 4.4
for i in range(6):
    y = LY + i * (LH + GAP)
    w(f'  <rect x="{LX}" y="{y}" width="{LW}" height="{LH}" rx="4" fill="#21262d" stroke="#30363d"/>')
    w(f'  <rect x="{LX + LW + 6}" y="{y + 2}" width="0" height="{LH - 4}" rx="3" fill="url(#lg)">'
      + anim("width", [(0, 0), (TRAIN0 + i * 0.12, 0), (TRAIN1, 46), (END, 46), (END + 0.3, 0), (T, 0)]) + '</rect>')
w(f'  <text class="t k" x="{LX + 16}" y="{LY - 8}">frozen weights</text>')
w(f'  <text class="t k" x="{LX + LW + 6}" y="{LY - 8}" opacity="0" fill="{GREEN}" style="fill:{GREEN}">adapter{show(TRAIN0)}</text>')
# a lock on the frozen weights
w(f'  <g transform="translate({LX} {LY - 19}) scale(0.85)" opacity="0.8"><rect x="0" y="5" width="12" height="9" rx="2" fill="none" stroke="#6e7681" stroke-width="1.4"/>'
  f'<path d="M3,5 V3 a3,3 0 0 1 6,0 V5" fill="none" stroke="#6e7681" stroke-width="1.4"/></g>')
# training: a bar and its time
BY = LY + 6 * (LH + GAP) + 4
w(f'  <rect x="{LX}" y="{BY}" width="{LW - 26}" height="6" rx="3" fill="#21262d"/>')
w(f'  <rect x="{LX}" y="{BY}" width="0" height="6" rx="3" fill="url(#lg)">'
  + anim("width", [(0, 0), (TRAIN0, 0), (TRAIN1, LW - 26), (END, LW - 26), (END + 0.3, 0), (T, 0)]) + '</rect>')
w(f'  <text class="t k" x="{LX}" y="{BY + 20}" opacity="0">training, ~25 min on the 8 GB GPU{show(TRAIN0)}</text>')

# the facts fly from the chats into the adapter (not the frozen weights), shrinking to its strips as they land;
# drawn after the layers so they pass over them
SHRINK = 0.32
for i, f in enumerate(facts):
    sx, sy = 46, TOP + 40 + (1 + 2 * i) * 34 + 3
    row = LY + 2 * i * (LH + GAP)
    tx, ty = LX + LW + 6, row + LH / 2 - 8 * SHRINK
    t0 = DAY_END + 0.1 + i * 0.25
    t1 = t0 + 1.0
    keys = f'keyTimes="0;{t0 / T:.4f};{t1 / T:.4f};1" dur="{T}s" repeatCount="indefinite" calcMode="spline" keySplines="0 0 1 1;0.5 0 0.3 1;0 0 1 1"'
    fw = chip_w(f)
    w(f'  <g opacity="0">{anim("opacity", [(0, 0), (t0, 0), (t0 + 0.1, 1), (t1 - 0.1, 1), (t1 + 0.2, 0), (T, 0)])}'
      f'<animateTransform attributeName="transform" type="translate" values="{sx} {sy};{sx} {sy};{tx} {ty:.1f};{tx} {ty:.1f}" {keys}/>'
      f'<g><animateTransform attributeName="transform" type="scale" values="1;1;{SHRINK};{SHRINK}" {keys}/>'
      f'<rect x="0" y="0" width="{fw:.0f}" height="16" rx="8" fill="{GREEN}"/><text class="t chip" x="8" y="12" textLength="{text_w(f):.1f}" lengthAdjust="spacingAndGlyphs">{esc(f)}</text></g></g>')
# the checks, then installed
checks = [("facts learned", "loss 6.36 → 0.66"), ("general answers kept", "no drift"), ("still calls its tools", "6 of 6")]
CY = BY + 38
for i, (what, how) in enumerate(checks):
    t0 = TRAIN1 + 0.3 + i * 0.45
    y = CY + i * 22
    w(f'  <g opacity="0">{show(t0)}<circle cx="{LX + 7}" cy="{y - 4}" r="7" fill="{GREEN}"/>'
      f'<path d="M{LX + 3.5},{y - 4} l2.5,2.5 l4.5,-5" fill="none" stroke="#0d1117" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
      f'<text class="t ok" x="{LX + 20}" y="{y}">{what}</text><text class="t k" x="{LX + 232}" y="{y}" text-anchor="end">{how}</text></g>')
INST = TRAIN1 + 0.3 + 3 * 0.45 + 0.2
w(f'''  <g opacity="0">{show(INST)}
    <rect x="{LX}" y="{CY + 56}" width="232" height="22" rx="11" fill="{GREEN}"/>
    <text class="t b" x="{LX + 116}" y="{CY + 71}" text-anchor="middle" fill="#0d1117">✓ installed: the 4B uses it from its next load</text>
  </g>''')

# ---- the morning: it knows, and it still uses its tools
MX = 602
Q1, A1, Q2, A2 = NIGHT_END + 0.3, NIGHT_END + 1.1, NIGHT_END + 2.2, NIGHT_END + 3.0
w(f'  <g opacity="0">{show(Q1)}<rect x="{MX + 60}" y="{TOP + 40}" width="160" height="26" rx="13" fill="#1c2430" stroke="#30363d"/>'
  f'<text class="t say" x="{MX + 74}" y="{TOP + 57}">When do we deploy?</text></g>')
w(f'''  <g opacity="0">{show(A1)}
    <g transform="translate({MX + 4} {TOP + 76}) scale(0.9)"><path d="{GHOST}" fill="{GREEN}"/><ellipse cx="10.4" cy="10.6" rx="1.5" ry="2.1" fill="#0f141b"/><ellipse cx="15.4" cy="10.6" rx="1.5" ry="2.1" fill="#0f141b"/></g>
    <rect x="{MX + 30}" y="{TOP + 76}" width="112" height="26" rx="13" fill="#12261f" stroke="{GREEN}"/>
    <text class="t say" x="{MX + 44}" y="{TOP + 93}">On Fridays.</text>
    <text class="t k" x="{MX + 30}" y="{TOP + 120}">from its weights: not in the prompt,</text>
    <text class="t k" x="{MX + 30}" y="{TOP + 135}">not in a notes file</text>
  </g>''')
w(f'  <g opacity="0">{show(Q2)}<rect x="{MX + 44}" y="{TOP + 154}" width="176" height="26" rx="13" fill="#1c2430" stroke="#30363d"/>'
  f'<text class="t say" x="{MX + 58}" y="{TOP + 171}">What CPU does this PC have?</text></g>')
w(f'''  <g opacity="0">{show(A2)}
    <g transform="translate({MX + 4} {TOP + 190}) scale(0.9)"><path d="{GHOST}" fill="{GREEN}"/><ellipse cx="10.4" cy="10.6" rx="1.5" ry="2.1" fill="#0f141b"/><ellipse cx="15.4" cy="10.6" rx="1.5" ry="2.1" fill="#0f141b"/></g>
    <rect x="{MX + 30}" y="{TOP + 190}" width="180" height="22" rx="11" fill="{GREEN}"/>
    <text class="m" x="{MX + 40}" y="{TOP + 205}" font-size="11" fill="#0d1117"><tspan font-weight="700">tool call</tspan> get_system_info</text>
    <text class="t k" x="{MX + 30}" y="{TOP + 232}">and it still uses its tools</text>
  </g>''')

w(f'  <text class="t k" x="24" y="{H - 16}">Qwen3-4B, trained on 21 facts from real chats. Not installed unless all three checks pass; the adapter before it is kept.</text>')
w("</svg>")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "sleep.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote sleep.svg")
