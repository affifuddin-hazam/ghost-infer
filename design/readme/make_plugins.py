"""Draws plugins.svg: the model makes its own tool. Asked for something it cannot do, it writes a tools.py,
ghost starts it as a plugin, and the new tool is called in the same chat. The code and the result are the
dice plugin Qwen3-4B wrote in the test on 2026-11-09. Run: python make_plugins.py   (writes plugins.svg)"""
import os

T = 14.0
END = T - 0.4
W, H = 860, 420
GREEN, LINE = "#7ee2b8", "#3fb68b"


def anim(attr, pairs, extra=""):
    k = ";".join(f"{p[0] / T:.4f}" for p in pairs)
    v = ";".join(str(p[1]) for p in pairs)
    return f'<animate attributeName="{attr}" values="{v}" keyTimes="{k}" dur="{T}s" repeatCount="indefinite"{extra}/>'


def window(t0, t1=None, top=1):
    if t1 is None:
        return [(0, 0), (t0, 0), (t0 + 0.15, top), (END, top), (END + 0.3, 0), (T, 0)]
    return [(0, 0), (t0, 0), (t0 + 0.15, top), (t1 - 0.15, top), (t1, 0), (T, 0)]


def shown(t0, t1=None, top=1):
    return anim("opacity", window(t0, t1, top))


def typed(cid, x, y, w, h, t0, dur):
    return (f'<clipPath id="{cid}"><rect x="{x}" y="{y}" height="{h}" width="0">'
            + anim("width", [(0, 0), (t0, 0), (t0 + dur, w), (END, w), (END + 0.01, 0), (T, 0)]) + "</rect></clipPath>")


out = []
w = out.append
w(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="ghost builds its own tools: asked to roll dice, it writes a small Python plugin, ghost starts it, and the new roll_dice tool is called in the same chat">
  <style>
    .t  {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .m  {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; }}
    .h  {{ font-size: 18px; font-weight: 700; fill: #e6edf3; }}
    .s  {{ font-size: 12px; fill: #8b949e; }}
    .k  {{ font-size: 11px; fill: #8b949e; }}
    .kw {{ fill: #c9d1d9; }}
    .code {{ font-size: 11px; fill: #c9d1d9; }}
    .kwd {{ fill: #ff7b72; }}
    .fn  {{ fill: #d2a8ff; }}
    .str {{ fill: #a5d6ff; }}
    .num {{ fill: #79c0ff; }}
    .chip {{ font-size: 11px; fill: #c9d1d9; }}
    .ask {{ font-size: 15px; fill: #e6edf3; font-style: italic; }}
  </style>
  <defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4"/></filter></defs>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="#0d1117" stroke="#30363d"/>
  <text class="t h" x="24" y="40">It builds its own tools</text>
  <text class="t s" x="24" y="62">Ask for something it cannot do yet: it writes a small plugin, ghost starts it, and the new tool works in the same chat.</text>
  <text class="t s" x="24" y="80">Each plugin runs as a program of its own, and every call to it asks you first.</text>''')

# ---- the request
w(f'  <text class="t ask" x="24" y="114" opacity="0">“Roll 3 dice with 20 sides for me.”{shown(0.2)}</text>')
w(f'  <text class="t k" x="290" y="114" opacity="0">no tool for that yet…{shown(0.9, 2.2)}</text>')

# ---- the code it writes
w('''  <rect x="24" y="128" width="356" height="232" rx="8" fill="#010409" stroke="#30363d"/>
  <path d="M24,150 H380" stroke="#30363d"/>
  <text class="m k" x="38" y="144">tools.py</text>''')
code = [
    '<tspan class="kwd">import</tspan> random',
    "",
    '<tspan class="kwd">def</tspan> <tspan class="fn">roll_dice</tspan>(count: int = <tspan class="num">2</tspan>,',
    '              sides: int = <tspan class="num">6</tspan>) -&gt; str:',
    '    <tspan class="str">"""Roll dice and give each roll and the total.</tspan>',
    '<tspan class="str">    count: how many dice</tspan>',
    '<tspan class="str">    sides: how many sides each die has</tspan>',
    '<tspan class="str">    """</tspan>',
    '    rolls = [random.randint(<tspan class="num">1</tspan>, sides)',
    '             <tspan class="kwd">for</tspan> _ <tspan class="kwd">in</tspan> range(count)]',
    '    <tspan class="kwd">return</tspan> <tspan class="str">f"rolls: {rolls}, total: {sum(rolls)}"</tspan>',
]
t = 1.8
for n, line in enumerate(code):
    y = 172 + n * 16
    if not line:
        t += 0.15
        continue
    dur = 0.12 + 0.012 * len(line.replace("<tspan", "").split(">")[-1]) if "<" in line else 0.12 + 0.012 * len(line)
    dur = min(max(dur, 0.25), 0.6)
    w("  " + typed(f"c{n}", 36, y - 12, 340, 16, t, dur))
    w(f'  <text class="m code" x="38" y="{y}" clip-path="url(#c{n})" xml:space="preserve">{line}</text>')
    t += dur + 0.05
code_done = t   # about 6 s

# ---- ghost starts it
w(f'''  <g opacity="0">{shown(code_done + 0.1)}
    <path d="M392,246 H468" fill="none" stroke="#6e7681" stroke-width="1.6" stroke-dasharray="5 4">{anim("stroke-dashoffset", [(0, 0), (T, -90)])}</path>
    <path d="M462,240 L469,246 L462,252" fill="none" stroke="#6e7681" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/>
  </g>''')
gx, gy = 430, 212
w(f'  <g opacity="0" transform="translate({gx} {gy})">{shown(code_done + 0.1)}')
w('    <g>' + "".join(f'<rect x="-3" y="-15" width="6" height="7" rx="1.5" fill="#6e7681" transform="rotate({a})"/>' for a in range(0, 360, 45))
  + f'<circle r="10" fill="#161b22" stroke="#6e7681" stroke-width="2.5"/><circle r="3.5" fill="#6e7681"/>'
  + f'<animateTransform attributeName="transform" type="rotate" values="0;0;360;360" keyTimes="0;{(code_done + 0.1) / T:.4f};{(code_done + 1.4) / T:.4f};1" dur="{T}s" repeatCount="indefinite" calcMode="spline" keySplines="0 0 1 1;0.4 0 0.3 1;0 0 1 1"/></g>')
w("  </g>")
w(f'  <text class="t k" x="{gx}" y="272" text-anchor="middle" opacity="0">ghost starts it{shown(code_done + 0.2)}</text>')
w(f'  <text class="t k" x="{gx}" y="288" text-anchor="middle" opacity="0" style="fill:{GREEN}">1 new tool{shown(code_done + 1.4)}</text>')

# ---- its tools
w('''  <rect x="480" y="128" width="356" height="232" rx="8" fill="#161b22" stroke="#30363d"/>
  <text class="t k kw" x="496" y="148" style="font-weight:700">its tools</text>''')
tools = ["read_file", "write_file", "edit_file", "list_files", "search_files", "run_command", "web_search", "fetch_url", "remember", "schedule_task"]
x, y = 496, 160
for name in tools:
    wd = 14 + 6.6 * len(name)
    if x + wd > 822:
        x, y = 496, y + 28
    w(f'  <rect x="{x}" y="{y}" width="{wd:.0f}" height="20" rx="10" fill="#0d1117" stroke="#30363d"/><text class="m chip" x="{x + 7}" y="{y + 14}">{name}</text>')
    x += wd + 6
# the new one
new_at = code_done + 1.4
nx, ny = (x, y) if x + 90 <= 822 else (496, y + 28)
w(f'''  <g opacity="0">{shown(new_at)}
    <rect x="{nx - 2}" y="{ny - 2}" width="88" height="24" rx="12" fill="{GREEN}" opacity="0" filter="url(#glow)">{anim("opacity", [(0, 0), (new_at, 0), (new_at + 0.15, 0.8), (new_at + 1.4, 0), (T, 0)])}</rect>
    <rect x="{nx}" y="{ny}" width="84" height="20" rx="10" fill="#1f3b33" stroke="{LINE}"/>
    <text class="m chip" x="{nx + 8}" y="{ny + 14}" style="fill:{GREEN}">roll_dice</text>
    <text class="t k" x="{nx + 90}" y="{ny + 14}" style="fill:{GREEN}">new</text>
  </g>''')

# ---- the call, in the same chat
call_at = new_at + 1.0
w(f'  <path d="M496,{ny + 36} H822" stroke="#30363d" opacity="0">{shown(call_at)}</path>')
cy = ny + 58
w("  " + typed("call", 494, cy - 13, 330, 18, call_at, 0.7))
w(f'  <text class="m code" x="496" y="{cy}" clip-path="url(#call)"><tspan class="fn">roll_dice</tspan>(count=<tspan class="num">3</tspan>, sides=<tspan class="num">20</tspan>)</text>')
w(f'  <text class="m code" x="496" y="{cy + 22}" opacity="0">→ <tspan style="fill:{GREEN}">rolls: [5, 12, 14], total: 31</tspan>{shown(call_at + 1.0)}</text>')

# ---- the steps
steps = [("you ask for something new", 0.2, 1.8), ("it writes tools.py", 1.8, code_done + 0.1),
         ("ghost starts it as a plugin", code_done + 0.1, call_at), ("it calls its new tool", call_at, None)]
for n, (label, t0, t1) in enumerate(steps):
    x = 24 + n * 206
    w(f'  <rect x="{x}" y="374" width="196" height="30" rx="15" fill="#161b22" stroke="#30363d"/>')
    w(f'  <rect x="{x}" y="374" width="196" height="30" rx="15" fill="#1f3b33" stroke="{LINE}" opacity="0">{shown(t0, t1)}</rect>')
    w(f'  <circle cx="{x + 16}" cy="389" r="8" fill="#30363d"/><text class="t k kw" x="{x + 16}" y="393" text-anchor="middle" style="font-weight:700">{n + 1}</text>')
    w(f'  <text class="t k kw" x="{x + 30}" y="393">{label}</text>')
w("</svg>")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "plugins.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote plugins.svg, code done at", round(code_done, 1), "s")
