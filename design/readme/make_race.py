"""race.svg: three engines write the same answer, replayed from the token times bench-moe.py recorded.

Each lane shows the engine's real streamed text, revealed at the moment each piece arrived (time from sending the
request, so the wait for the first token counts), with the progress bar following the same clock.

  make_race.py [--ghost F] [--llama F] [--ollama F] [--prompt explain] [--round 1]
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

X0, W = 24, 812          # lane left edge and width
CW = 7.25                # px per character of the streamed text (forced with textLength, so fonts can't drift)
LANE_Y = (198, 270, 342)  # baseline of each lane's name


# ghost-infer's mark: one flat shape (dome, three-scallop hem), eyes cut out and glancing right, down the track
GHOST = ("M4,11 C4,6.2 7.6,2.5 12,2.5 C16.4,2.5 20,6.2 20,11 L20,21 Q18.67,18.6 17.33,21 Q16,23.4 14.67,21 "
         "Q13.33,18.6 12,21 Q10.67,23.4 9.33,21 Q8,18.6 6.67,21 Q5.33,23.4 4,21 Z")
GHOST_EYES = ((10.4, 10.6), (15.4, 10.6))

# the block letters of the GPU's "RTX 2060" (14 x 22 cells, rounded by 4), extended to spell GHOST-INFER
GLYPHS = {
    "G": (14, "M14,4 Q14,0 10,0 H4 Q0,0 0,4 V18 Q0,22 4,22 H10 Q14,22 14,18 V11 H7"),
    "H": (14, "M0,0 V22 M14,0 V22 M0,11 H14"),
    "O": (14, "M4,0 H10 Q14,0 14,4 V18 Q14,22 10,22 H4 Q0,22 0,18 V4 Q0,0 4,0 Z"),
    "S": (14, "M14,4 Q14,0 10,0 H4 Q0,0 0,4 V7 Q0,11 4,11 H10 Q14,11 14,15 V18 Q14,22 10,22 H4 Q0,22 0,18"),
    "T": (14, "M0,0 H14 M7,0 V22"),
    "-": (10, "M1,11 H9"),
    "I": (8, "M0,0 H8 M4,0 V22 M0,22 H8"),
    "N": (14, "M0,22 V0 L14,22 V0"),
    "F": (14, "M14,0 H0 V22 M0,11 H10"),
    "E": (14, "M14,0 H0 V22 H14 M0,11 H10"),
    "R": (14, "M0,22 V0 H10 Q14,0 14,4 V7 Q14,11 10,11 H0 M8,11 L14,22"),
}


def wordmark(word, x, y, scale=0.5, color="#7ee2b8", core="#0d1117"):
    """`word` in the GPU's lettering, cap tops at y: a thick colour stroke with a thin dark one down its middle."""
    paths, at = [], 0
    for ch in word:
        w, d = GLYPHS[ch]
        paths.append(f'<path transform="translate({at} 0)" d="{d}"/>')
        at += w + 6
    p = "".join(paths)
    return (f'<g transform="translate({x} {y}) scale({scale})" fill="none" stroke-linecap="square" stroke-linejoin="miter" stroke-miterlimit="2">'
            f'<g stroke="{color}" stroke-width="5.2">{p}</g><g stroke="{core}" stroke-width="2.2">{p}</g></g>'), (at - 6) * scale


# "infer" as a solid tag in the manner of a graphics card's model badge: a filled slab, square left end, slanted right
# end, heavy slanted lowercase cut out of it; the i's foot runs out to the left edge, the r's arm out to the right.
# Units: the tag is 30 high, letters sit on y=25 with a 13-unit x-height; everything is clipped to the tag.
TAG_W = 80
TAG = f"M0,0 H{TAG_W} L{TAG_W - 9},30 H0 Z"
INFER = ("M-10,25 H4.5 V12"                                   # i, with its foot out to the left edge
         " M4.5,4 V8.4"                                    # i's dot
         " M11,25 V12 H19 Q23,12 23,16 V25"                     # n
         " M30,40 V8 Q30,4 34,4 H38 M26,12.5 H36.5"             # f, its stem out through the bottom edge
         " M41,18.5 H53.5 V15.5 Q53.5,12 50,12 H44.5 Q41,12 41,15.5 V21.5 Q41,25 44.5,25 H53.5"  # e, one stroke
         " M59,25 V16 Q59,12 63,12 H95")                        # r, its arm out to the right edge


# the tag's size, tuned by eye with tune.html: its height in px, its letters' size inside it, and their weight
TAG_H, TAG_LETTERS, TAG_WEIGHT = 11.25, 0.82, 4.4
LETTERS_AT = (32, 18.5)  # the letters grow and shrink about this point (tag units)


def infer_tag(x, mid, color="#7ee2b8", ink="#0d1117", cid="tagclip"):
    """The tag, `TAG_H` px tall and vertically centred on `mid`. tune.html finds it by its ids."""
    s = TAG_H / 30
    cx, cy = LETTERS_AT
    return (f'<g id="infer-tag" data-mid="{mid:.2f}" transform="translate({x:.1f} {mid - TAG_H / 2:.2f}) scale({s:.4f})">'
            f'<clipPath id="{cid}"><path d="{TAG}"/></clipPath>'
            f'<path d="{TAG}" fill="{color}"/>'
            f'<g clip-path="url(#{cid})"><g id="infer-letters" transform="translate({cx} {cy}) scale({TAG_LETTERS}) '
            f'translate({-cx} {-cy})"><path transform="translate(8.5 0) skewX(-14)" d="{INFER}" '
            f'fill="none" stroke="{ink}" stroke-width="{TAG_WEIGHT}" stroke-linecap="butt" stroke-linejoin="miter"/>'
            f'</g></g></g>')


# GHOST in GeForce Bold Alt, outlined once into ghost_word.svgpart (font units: baseline 0, ink x 123..6667, caps 1397)
GHOST_WORD = re.search(r'd="([^"]+)"', (HERE / "ghost_word.svgpart").read_text(encoding="utf-8")).group(1)


def logotype(x, y, color="#7ee2b8", cap=11):
    """GHOST, solid, cap tops at y and `cap` px tall; then the "infer" tag, a little taller and centred on it."""
    s = cap / 1397
    word = (f'<path transform="translate({x - 123 * s:.2f} {y + cap}) scale({s:.5f})" d="{GHOST_WORD}" '
            f'fill="{color}"/>')
    return word + infer_tag(x + (6667 - 123) * s + 5, y + cap / 2, color)


def load(path, prompt, rnd):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for r in d["requests"]:
        if r["counted"] and r["prompt"] == prompt and r["round"] == rnd:
            return r
    raise SystemExit(f"{path}: no round {rnd} of {prompt!r}")


def track(r):
    """(seconds, pieces so far, characters so far) at each arrival, and the cleaned text."""
    text, pts = "", [(0.0, 0, 0)]
    for t, piece in zip(r["arrivals"], r["pieces"]):
        piece = re.sub(r"\s+", " ", piece.replace("<think>", "").replace("</think>", ""))
        if text.endswith(" ") or not text:
            piece = piece.lstrip()
        text += piece
        pts.append((t, len(pts), len(text)))
    if pts[1][0] > 0.01:  # nothing shows until the first piece
        pts.insert(1, (pts[1][0] - 0.01, 0, 0))
    return pts, text


def anim(attr, pts, dur, f, extra=""):
    """An <animate> through (t, value) points, holding the last value to the end of the loop."""
    keys, vals = [], []
    for t, v in [(t, f(p, c)) for t, p, c in pts] + [(dur, None)]:
        k = round(t / dur, 4)
        v = vals[-1] if v is None else v
        if keys and k <= keys[-1]:
            vals[-1] = v
            continue
        keys.append(k)
        vals.append(v)
    return (f'<animate attributeName="{attr}" values="{";".join(vals)}" keyTimes="{";".join(map(str, keys))}" '
            f'dur="{dur}s" repeatCount="indefinite"{extra}/>')


def lane(i, name, logo, r, dur, lead):
    y = LANE_Y[i]
    pts, text = track(r)
    n = pts[-1][1]
    end = pts[-1][0]
    tps = r["gen_tps"]
    g = "ng" if lead else ""
    out = [f"  <!-- {name}: {r['gen_tokens']} tokens, {tps:.1f} tok/s, done {end:.2f} s after sending -->",
           logo(y),
           ("  " + logotype(X0 + 27, y - 11) if lead else
            f'  <text class="t n {g}" x="{X0 + 26}" y="{y}">{name}</text>'),
           f'  <text class="t v {"vg" if lead else ""}" x="{X0 + W}" y="{y}">{tps:.1f} tok/s</text>',
           f'  <rect x="{X0}" y="{y + 8}" width="{W}" height="8" rx="4" fill="#21262d"/>',
           f'  <rect x="{X0}" y="{y + 8}" width="0" height="8" rx="4" fill="{"url(#g)" if lead else "#6e7681"}">',
           "    " + anim("width", pts, dur, lambda p, c: f"{W * p / n:.1f}"),
           "  </rect>"]
    if lead:
        out += [f'  <circle cx="{X0}" cy="{y + 12}" r="6" fill="#7ee2b8" filter="url(#glow)">',
                "    " + anim("cx", pts, dur, lambda p, c: f"{X0 + W * p / n:.1f}"),
                f'    <animate attributeName="opacity" values="1;1;0;0" keyTimes="0;{end / dur:.4f};{min(end / dur + 0.02, 0.99):.4f};1" '
                f'dur="{dur}s" repeatCount="indefinite"/>',
                "  </circle>"]
    # the text: a clip that grows with it, and once it is wider than the lane, the text slides left
    clip = f"c{i}"
    out += [f'  <clipPath id="{clip}"><rect x="{X0}" y="{y + 24}" height="20" width="0">',
            "    " + anim("width", pts, dur, lambda p, c: f"{min(c * CW, W):.1f}"),
            "  </rect></clipPath>",
            f'  <g clip-path="url(#{clip})"><g>',
            "    " + anim("transform", pts, dur, lambda p, c: f"{-max(0.0, c * CW - W):.1f} 0",
                          "").replace("<animate ", '<animateTransform type="translate" '),
            f'    <text class="m c {"cg" if lead else ""}" x="{X0}" y="{y + 38}" textLength="{len(text) * CW:.1f}" '
            f'lengthAdjust="spacing">{esc(text)}</text>',
            "  </g></g>"]
    a, b = end / dur, min(end / dur + 0.02, 0.96)
    label = f"✓ {end:.1f} s"
    wpx = 62 if end < 10 else 70
    out += ['  <g opacity="0">',
            f'    <animate attributeName="opacity" values="0;0;1;1;0" keyTimes="0;{a:.4f};{b:.4f};0.97;1" dur="{dur}s" '
            'repeatCount="indefinite"/>',
            f'    <rect x="{734 - wpx}" y="{y - 14}" width="{wpx}" height="20" rx="10" fill="{"#7ee2b8" if lead else "#30363d"}"/>',
            f'    <text class="t b" x="{734 - wpx / 2:.0f}" y="{y}" text-anchor="middle" fill="{"#0d1117" if lead else "#c9d1d9"}">'
            f"{label}</text>",
            "  </g>", ""]
    return "\n".join(out), end


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main():
    global TAG_H, TAG_LETTERS, TAG_WEIGHT
    p = argparse.ArgumentParser()
    p.add_argument("--ghost", default=ROOT / "bench-readme-20261114-081509" / "moe" / "ghost-infer.json")
    p.add_argument("--llama", default=ROOT / "bench-readme-20261114-081509" / "moe" / "llama.cpp.json")
    p.add_argument("--ollama", default=ROOT / "bench-readme-20261114-081509" / "moe" / "ollama.json")
    p.add_argument("--prompt", default="explain")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--model", default="Qwen3.6-35B-A3B Q3_K_M, a 16.8 GB model")
    p.add_argument("--versions", default="llama.cpp b11747 (--n-cpu-moe 24), Ollama 0.40.13")
    p.add_argument("--tag-h", type=float, help=f"the infer tag's height in px (default {TAG_H})")
    p.add_argument("--tag-letters", type=float, help=f"its letters' size inside it, 1 = as drawn (default {TAG_LETTERS})")
    p.add_argument("--tag-weight", type=float, help=f"its letters' stroke, tag units (default {TAG_WEIGHT})")
    a = p.parse_args()
    TAG_H = a.tag_h or TAG_H
    TAG_LETTERS = a.tag_letters or TAG_LETTERS
    TAG_WEIGHT = a.tag_weight or TAG_WEIGHT

    runs = [load(f, a.prompt, a.round) for f in (a.ghost, a.llama, a.ollama)]
    last = max(r["arrivals"][-1] for r in runs)
    dur = int(last + 4.5)

    eyes = "".join(f'<ellipse cx="{x}" cy="{ey}" rx="1.5" ry="2.1" fill="#0d1117"/>' for x, ey in GHOST_EYES)
    logos = [
        lambda y: f'  <g transform="translate({X0 - 2} {y - 17}) scale(0.92)"><path d="{GHOST}" fill="#7ee2b8"/>{eyes}</g>',
        lambda y: f'  <use href="#llamacpp" x="{X0}" y="{y - 15}" width="18" height="18" fill="#8b949e"/>',
        lambda y: f'  <use href="#ollama" x="{X0 + 3}" y="{y - 15}" width="14" height="19" fill="#8b949e"/>',
    ]
    names = ["ghost-infer", "llama.cpp", "Ollama"]
    lanes, ends = [], []
    for i, (name, r) in enumerate(zip(names, runs)):
        s, end = lane(i, name, logos[i], r, dur, i == 0)
        lanes.append(s)
        ends.append(end)

    k = last / dur
    fan = (lambda c: f'values="0 {c} 81;5400 {c} 81;5490 {c} 81;5490 {c} 81" keyTimes="0;{k:.3f};{min(k + 0.13, 0.99):.3f};1"')
    gpu = (HERE / "race_gpu.svgpart").read_text(encoding="utf-8")
    gpu = gpu.replace("{FAN92}", fan(92)).replace("{FAN252}", fan(252)).replace('dur="15s"', f'dur="{dur}s"')
    ollama = (HERE / "logo_ollama.svgpart").read_text(encoding="utf-8") + \
        (HERE / "logo_llamacpp.svgpart").read_text(encoding="utf-8")
    t = [f"{n} {r['gen_tps']:.1f}" for n, r in zip(names, runs)]

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 422" width="860" height="422" role="img" aria-label="Token race on {esc(a.model)}, RTX 2060 SUPER, tokens per second: {", ".join(t)}">
  <!-- made by make_race.py from bench-moe.py's recorded token times; edit that, not this -->
  <style>
    .t  {{ font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .m  {{ font-family: ui-monospace, "Cascadia Code", Consolas, "SFMono-Regular", monospace; }}
    .h  {{ font-size: 18px; font-weight: 700; fill: #e6edf3; }}
    .s  {{ font-size: 12px; fill: #8b949e; }}
    .n  {{ font-size: 14px; font-weight: 600; fill: #8b949e; }}
    .ng {{ fill: #7ee2b8; }}
    .v  {{ font-size: 14px; font-weight: 700; fill: #8b949e; text-anchor: end; }}
    .vg {{ fill: #7ee2b8; }}
    .c  {{ font-size: 12px; fill: #6e7681; }}
    .cg {{ fill: #c9d1d9; }}
    .b  {{ font-size: 11px; font-weight: 700; }}
    .ln   {{ fill: none; stroke: #6e7681; stroke-width: 1.4; }}
    .body {{ fill: #161b22; stroke: #8b949e; stroke-width: 1.6; }}
    .bl   {{ fill: #1c232c; stroke: #8b949e; stroke-width: 1.2; stroke-linejoin: round; }}
    .hub  {{ fill: #0d1117; stroke: #8b949e; stroke-width: 1.4; }}
    .fin  {{ stroke: #484f58; stroke-width: 1.2; }}
    .dot  {{ fill: #6e7681; }}
  </style>
  <defs>
    <linearGradient id="g" x1="0" x2="1">
      <stop offset="0" stop-color="#3fb68b"/>
      <stop offset="1" stop-color="#7ee2b8"/>
    </linearGradient>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3"/>
    </filter>
    {ollama.strip()}
  </defs>

  <rect x="0.5" y="0.5" width="859" height="421" rx="12" fill="#0d1117" stroke="#30363d"/>

  <text class="t h" x="24" y="44">Same prompt, same GPU:</text>
  <text class="t h" x="24" y="68">who finishes 256 tokens first?</text>
  <text class="t s" x="24" y="100">{esc(a.model)}</text>
  <text class="t s" x="24" y="118">on an RTX 2060 SUPER: 8 GB, 448 GB/s, from 2019</text>
  <text class="t s" x="24" y="136">so most of the model lives in system RAM</text>
  <text class="t s" x="24" y="160" font-style="italic">replayed from each engine's recorded token times</text>
{gpu.rstrip()}

{chr(10).join(lanes)}
  <text class="t s" x="24" y="408">{esc(a.versions)} · the model's own thinking, as each engine streamed it</text>
</svg>
'''
    (HERE / "race.svg").write_text(svg, encoding="utf-8")
    print(f"race.svg: {len(svg.encode()) / 1024:.0f} KB, loop {dur} s; " + ", ".join(
        f"{n} done {e:.2f} s" for n, e in zip(names, ends)))


if __name__ == "__main__":
    main()
