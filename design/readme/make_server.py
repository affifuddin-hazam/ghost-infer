"""Draws server.svg: a drop-in server.
Four kinds of client point at the same /v1 address; one after another each sends a request, ghost-infer streams
the reply back, and the agent's reply is a tool call. Below: the one line of client code that changes.
Abstract boxes only: no real app's UI.
Run: python make_server.py   (writes server.svg next to it)"""
import os

T = 13.0
W, H = 860, 404
GREEN = "#7ee2b8"
END = T - 0.4

# the ghost mark, as in make_race.py
GHOST = ("M4,11 C4,6.2 7.6,2.5 12,2.5 C16.4,2.5 20,6.2 20,11 L20,21 Q18.67,18.6 17.33,21 Q16,23.4 14.67,21 "
         "Q13.33,18.6 12,21 Q10.67,23.4 9.33,21 Q8,18.6 6.67,21 Q5.33,23.4 4,21 Z")

CX, CW, CH = 24, 260, 46      # client boxes
SX, SY, SW, SH = 560, 96, 276, 196  # the server
CLIENTS = [
    # name, what it is set to, what comes back (shown once the reply has streamed), is it a tool call
    ("editor plugin", "API base  http://127.0.0.1:8080/v1", "def is_prime(n): ...", False),
    ("chat app", "provider  OpenAI-compatible, local", "Day 1: Georgetown, char kway teow at…", False),
    ("agent harness", "tools  read_file, edit_file, run", 'read_file("src/main.rs")', True),
    ("your script", "curl …/v1/chat/completions", '{"choices": [{"delta": {"content": "Hi"}}]}', False),
]
START, EACH = 0.5, 2.9  # client i sends at START + i*EACH
GO, BACK = 0.55, 1.3    # request travel time; reply streaming time


def anim(attr, pairs, extra=""):
    k = ";".join(f"{p[0] / T:.4f}" for p in pairs)
    v = ";".join(str(p[1]) for p in pairs)
    return f'<animate attributeName="{attr}" values="{v}" keyTimes="{k}" dur="{T}s" repeatCount="indefinite"{extra}/>'


def pulse(t0, t1, lo=0, hi=1):
    return anim("opacity", [(0, lo), (t0, lo), (t0 + 0.15, hi), (t1, hi), (t1 + 0.25, lo), (T, lo)])


def motion(path_id, t0, t1, forward=True):
    a, b = (0, 1) if forward else (1, 0)
    return (f'<animateMotion dur="{T}s" repeatCount="indefinite" calcMode="linear" keyPoints="{a};{a};{b};{b}" '
            f'keyTimes="0;{t0 / T:.4f};{t1 / T:.4f};1"><mpath href="#{path_id}"/></animateMotion>')


out = []
w = out.append
w(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="A drop-in server: editor plugins, chat apps, agent harnesses and scripts that speak the OpenAI API point at ghost-infer's /v1 address and get streamed replies and tool calls">
  <style>
    .t  {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .m  {{ font-family: ui-monospace, "Cascadia Code", Consolas, monospace; }}
    .h  {{ font-size: 18px; font-weight: 700; fill: #e6edf3; }}
    .s  {{ font-size: 12px; fill: #8b949e; }}
    .n  {{ font-size: 13px; font-weight: 600; fill: #c9d1d9; }}
    .c  {{ font-size: 10.5px; fill: #6e7681; }}
    .r  {{ font-size: 11px; fill: #c9d1d9; }}
    .ep {{ font-size: 12px; fill: #c9d1d9; }}
    .k  {{ font-size: 11px; fill: #8b949e; }}
    .code {{ font-size: 12.5px; fill: #c9d1d9; }}
    .gr {{ fill: {GREEN}; }}
    .dim {{ fill: #6e7681; }}
  </style>
  <defs><filter id="glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3"/></filter></defs>
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="12" fill="#0d1117" stroke="#30363d"/>
  <text class="t h" x="24" y="40">A drop-in server</text>
  <text class="t s" x="24" y="62">It speaks the OpenAI API, with streaming and tool calls: what already talks to that API works unchanged.</text>''')

# ---- the server
w(f'''  <rect x="{SX}" y="{SY}" width="{SW}" height="{SH}" rx="12" fill="#161b22" stroke="#30363d"/>
  <rect x="{SX}" y="{SY}" width="{SW}" height="{SH}" rx="12" fill="none" stroke="{GREEN}" stroke-width="1.6" opacity="0">'''
  + anim("opacity", [(0, 0)] + [p for i in range(4) for p in (
      (START + i * EACH + GO - 0.05, 0), (START + i * EACH + GO + 0.1, 1), (START + i * EACH + GO + BACK, 1),
      (START + i * EACH + GO + BACK + 0.3, 0))] + [(T, 0)]) + f'''</rect>
  <g transform="translate({SX + 20} {SY + 18}) scale(1.25)"><path d="{GHOST}" fill="{GREEN}"/>
    <ellipse cx="10.4" cy="10.6" rx="1.5" ry="2.1" fill="#161b22"/><ellipse cx="15.4" cy="10.6" rx="1.5" ry="2.1" fill="#161b22"/></g>
  <text class="t n" x="{SX + 56}" y="{SY + 36}">ghost-infer</text>
  <text class="m k" x="{SX + 56}" y="{SY + 52}">http://127.0.0.1:8080</text>
  <line x1="{SX + 16}" y1="{SY + 70}" x2="{SX + SW - 16}" y2="{SY + 70}" stroke="#30363d"/>
  <text class="m ep" x="{SX + 20}" y="{SY + 96}"><tspan class="gr">POST</tspan> /v1/chat/completions</text>
  <text class="m ep" x="{SX + 20}" y="{SY + 118}"><tspan class="gr">GET </tspan> /v1/models</text>''')
for j, chipname in enumerate(["streaming", "tool calls"]):
    cw = len(chipname) * 6.4 + 20
    cx = SX + 20 + sum(len(c) * 6.4 + 28 for c in ["streaming", "tool calls"][:j])
    w(f'  <rect x="{cx:.0f}" y="{SY + 140}" width="{cw:.0f}" height="22" rx="11" fill="none" stroke="#30363d"/>'
      f'<text class="t k" x="{cx + cw / 2:.0f}" y="{SY + 155}" text-anchor="middle">{chipname}</text>')

# ---- the clients and their wires
for i, (name, setting, reply, tool) in enumerate(CLIENTS):
    y = 92 + i * 54
    cy = y + CH / 2
    t0 = START + i * EACH
    arrive = t0 + GO
    done = arrive + BACK
    pid = f"p{i}"
    ty = SY + 40 + i * 38  # where the wire meets the server
    w(f'  <path id="{pid}" d="M{CX + CW},{cy} C{CX + CW + 120},{cy} {SX - 120},{ty} {SX},{ty}" fill="none" stroke="#21262d" stroke-width="1.4"/>')
    w(f'  <path d="M{CX + CW},{cy} C{CX + CW + 120},{cy} {SX - 120},{ty} {SX},{ty}" fill="none" stroke="{GREEN}" stroke-width="1.4" opacity="0">'
      + pulse(t0, done, 0, 0.55) + '</path>')
    # the box, lit while it is waiting for its reply
    w(f'  <rect x="{CX}" y="{y}" width="{CW}" height="{CH}" rx="8" fill="#161b22" stroke="#30363d"/>')
    w(f'  <rect x="{CX}" y="{y}" width="{CW}" height="{CH}" rx="8" fill="none" stroke="{GREEN}" stroke-width="1.4" opacity="0">'
      + pulse(t0, done) + '</rect>')
    w(f'  <text class="t n" x="{CX + 14}" y="{y + 19}">{name}</text>')
    w(f'  <text class="m c" x="{CX + 14}" y="{y + 36}" opacity="1">{setting}'
      + anim("opacity", [(0, 1), (done - 0.05, 1), (done + 0.1, 0), (END, 0), (END + 0.2, 1), (T, 1)]) + '</text>')
    # the reply, typed out once it has streamed in
    clip = f"rc{i}"
    w(f'  <clipPath id="{clip}"><rect x="{CX + 10}" y="{y + 26}" height="16" width="0">'
      + anim("width", [(0, 0), (done - 0.05, 0), (done + 0.6, CW - 20), (END, CW - 20), (END + 0.01, 0), (T, 0)]) + '</rect></clipPath>')
    if tool:
        tw = len('tool call ' + reply) * 6.3 + 16
        w(f'  <g clip-path="url(#{clip})"><rect x="{CX + 12}" y="{y + 27}" width="{tw:.0f}" height="15" rx="7.5" fill="{GREEN}"/>'
          f'<text class="m" x="{CX + 20}" y="{y + 38}" font-size="10.5" fill="#0d1117"><tspan font-weight="700">tool call</tspan> {reply.replace(chr(34), "&quot;")}</text></g>')
    else:
        w(f'  <text class="m r" x="{CX + 14}" y="{y + 38}" clip-path="url(#{clip})">{reply.replace(chr(34), "&quot;")}</text>')
    # the request: one dot out
    w(f'  <circle r="4" fill="{GREEN}" filter="url(#glow)" opacity="0">{motion(pid, t0, arrive)}'
      + anim("opacity", [(0, 0), (t0, 0), (t0 + 0.05, 1), (arrive, 1), (arrive + 0.05, 0), (T, 0)]) + '</circle>')
    # the reply: a stream of small dots back (a pill for the tool call)
    for d in range(6):
        s0 = arrive + 0.05 + d * 0.13
        s1 = s0 + BACK - 0.6
        shape = (f'<rect x="-11" y="-5" width="22" height="10" rx="5" fill="{GREEN}"' if tool and d == 0 else
                 f'<circle r="2.4" fill="{GREEN}"')
        if tool and d > 0:
            break
        w(f'  {shape} opacity="0">{motion(pid, s0, s1, forward=False)}'
          + anim("opacity", [(0, 0), (s0, 0), (s0 + 0.05, 1), (s1, 1), (s1 + 0.05, 0), (T, 0)])
          + ('</rect>' if tool and d == 0 else '</circle>'))

# ---- the one line that changes
w('  <rect x="24" y="314" width="812" height="72" rx="8" fill="#010409" stroke="#30363d"/>')
w('  <text class="m code" x="40" y="340"><tspan class="dim">from</tspan> openai <tspan class="dim">import</tspan> OpenAI</text>')
w(f'  <rect x="36" y="352" width="586" height="22" rx="4" fill="{GREEN}" opacity="0.12"/>')
w(f'  <text class="m code" x="40" y="368">client = OpenAI(<tspan class="gr">base_url="http://127.0.0.1:8080/v1"</tspan>, api_key="none")</text>')
w(f'  <text class="t k" x="820" y="368" text-anchor="end">the only line that changes</text>')
w("</svg>")

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "server.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote server.svg")
