"""Speed of an OpenAI-compatible server on work that copies from its context, and on work that does not.

Agents spend much of their output on text they were given: a file written back with a change, a tool call
carrying a file, an identifier renamed through a file, quoted lines. Drafting by lookup (ghost-infer's
default) and llama.cpp's n-gram speculation speed that up; on fresh writing they can only cost time. So
there are two sets of prompts, sent through /v1/chat/completions, streamed, greedy, thinking off:

  copy-heavy: edit a file, rename through a file, a write_file tool call, fix and show code, convert data,
              quote a document
  new text:   a story, an explanation, a chat, new code, a word problem

  bench-lookup.py run --url http://127.0.0.1:8090 --model m --label ghost-infer --out ghost-infer.json
  bench-lookup.py summary <folder with the .json files> <label>[=<column title>] ...   (also writes summary.md)

Per request: generation = (completion tokens - 1) / (time from the first to the last streamed token). The
replies are kept, so `summary` can check that labels which must agree (ghost-infer with and without lookup)
wrote the same text.
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def section(text, heading, chars):
    """A README section (from its heading to the next one), cut to `chars`."""
    start = text.find(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:end if end > 0 else None][:chars]


def records():
    """A small made-up data set, the same every run."""
    cities = ["Penang", "Ipoh", "Kuching", "Melaka", "Johor Bahru", "Kota Kinabalu", "Shah Alam", "Seremban"]
    names = ["Aisyah", "Wei Jie", "Arun", "Nurul", "Daniel", "Mei Ling", "Hafiz", "Priya", "Siti", "Jason"]
    return [{"id": 1000 + i, "name": names[i % len(names)], "city": cities[(3 * i) % len(cities)],
             "score": round(50 + (i * 37) % 50 + (i % 3) * 0.5, 1), "active": i % 4 != 0} for i in range(18)]


def prompts():
    schedule = (ROOT / "src" / "server" / "schedule.rs").read_text(encoding="utf-8")
    sampler = (ROOT / "src" / "sampler.rs").read_text(encoding="utf-8")
    sampler = sampler[:sampler.find("#[cfg(test)]")]
    tokenizer = (ROOT / "src" / "tokenizer.rs").read_text(encoding="utf-8")[:3500]
    readme = section((ROOT / "README.md").read_text(encoding="utf-8"), "## How it works", 4000)
    code = schedule[:4000]
    copy = [
        ("edit a file", "Add a short `///` doc comment above each `fn` in this Rust code that has none, change "
                        "nothing else, and reply with the whole code, changed, in one code block.\n\n```rust\n"
                        + code + "\n```"),
        ("rename through a file", "Rename `cand` to `candidates` and `mass` to `kept_mass` everywhere in this file. "
                                  "Reply with the whole file in one code block and nothing else.\n\n```rust\n"
                                  + sampler + "\n```"),
        ("write_file tool call", "You are a coding agent. To change a file you reply with exactly one tool call: "
                                 "<tool_call>{\"name\": \"write_file\", \"arguments\": {\"path\": \"...\", "
                                 "\"content\": \"...\"}}</tool_call> where content is the complete new file. Set "
                                 "OK_MAX_CHARS to 500 in src/server/schedule.rs, which is:\n\n" + code),
        ("fix and show code", "Find one real bug or risky spot in this code, say what it is in one sentence, then "
                              "show the corrected code in full.\n\n```rust\n" + tokenizer + "\n```"),
        ("convert data", "Convert this JSON to a Markdown table with the same columns, then to CSV.\n\n"
                         + json.dumps(records(), indent=1)),
        ("quote a document", "Which settings, options or environment variables does this text mention? For each, "
                             "quote the sentence that mentions it, word for word.\n\n" + readme),
    ]
    fresh = [
        ("story", "Write a 400-word short story about a lighthouse keeper who finds a message in a bottle."),
        ("explain", "Explain how TCP's three-way handshake works, what SYN flooding is, and how SYN cookies defend "
                    "against it. Use an example with sequence numbers."),
        ("chat", "I'm planning a 3-day trip to Penang on a budget. Suggest an itinerary with food spots, rough costs "
                 "in MYR, and how to get around without a car."),
        ("write new code", "Write a Python function that reads a CSV file and returns, for each numeric column, the "
                           "mean, median and standard deviation, ignoring missing values. Then write three pytest "
                           "tests for it."),
        ("word problem", "A shop sells pens at RM2.50 each and notebooks at RM4.20 each. Aisyah buys 3 more pens than "
                         "notebooks and pays RM34.10 in total. How many of each did she buy? Show your working, then "
                         "check the answer."),
    ]
    return [("copy-heavy", n, t) for n, t in copy] + [("new text", n, t) for n, t in fresh]


def one(url, model, text, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    parts, usage = [], None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: {"):
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for c in d.get("choices") or []:
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    now = time.perf_counter()
                    parts.append(piece)
                    first = first or now
                    last = now
    gen = (usage or {}).get("completion_tokens") or len(parts)
    return {
        "ttft_s": (first - t0) if first else None,
        "gen_tokens": gen,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "gen_tps": (gen - 1) / (last - first) if first and last and last > first and gen > 1 else None,
        "text": "".join(parts),
    }


def run(a):
    ps = prompts()
    if a.prompts:
        # a smoke test: the first few of each set
        ps = [p for p in ps if p[0] == "copy-heavy"][: a.prompts] + [p for p in ps if p[0] == "new text"][: a.prompts]
    out = {"label": a.label, "rounds": a.rounds, "warmup": a.warmup, "max_tokens": a.max_tokens, "requests": []}
    for rnd in range(a.warmup + a.rounds):
        counted = rnd >= a.warmup
        for k, (kind, name, text) in enumerate(ps):
            # a label of its own in front, so no engine reads a prompt it already has in a cache from an earlier round
            r = one(a.url, a.model, f"[{rnd + 1}.{k + 1}] {text}", a.max_tokens)
            r.update(kind=kind, prompt=name, round=rnd - a.warmup + 1 if counted else 0, counted=counted)
            out["requests"].append(r)
            tag = f"round {r['round']}" if counted else "warm-up"
            gen = f"{r['gen_tps']:.1f}" if r["gen_tps"] else "-"
            print(f"   {a.label}, {tag}, {name}: {r['gen_tokens']} tokens, {gen} tok/s", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


def fmt(x, digits=1):
    return "-" if x is None else f"{x:,.{digits}f}"


def summary(folder, specs, same):
    # each spec is a result file's label, or label=column title
    labels = [s.split("=", 1)[0] for s in specs]
    titles = {s.split("=", 1)[0]: s.split("=", 1)[-1] for s in specs}
    data = {l: json.loads((Path(folder) / f"{l}.json").read_text(encoding="utf-8")) for l in labels}
    counted = {l: [r for r in d["requests"] if r["counted"]] for l, d in data.items()}
    med = lambda xs: statistics.median(xs) if xs else None
    head = ["| | " + " | ".join(titles[l] for l in labels) + " |", "|---|" + "---|" * len(labels)]
    lines = list(head)
    for kind in ("copy-heavy", "new text"):
        gen = {l: [r["gen_tps"] for r in rs if r["gen_tps"] and r["kind"] == kind] for l, rs in counted.items()}
        best = max((med(v) or 0) for v in gen.values())
        cell = lambda l: (f"**{fmt(med(gen[l]))}**" if med(gen[l]) == best else fmt(med(gen[l]))) + \
            (f" ({fmt(min(gen[l]), 0)}–{fmt(max(gen[l]), 0)})" if gen[l] else "")
        lines.append(f"| {kind.capitalize()}, tokens/s: median (range) | " + " | ".join(cell(l) for l in labels) + " |")
    lines += ["", "Per prompt (median over the measured rounds), tokens/s:", ""] + head
    for kind, name, _ in prompts():
        row = [med([r["gen_tps"] for r in counted[l] if r["prompt"] == name and r["gen_tps"]]) for l in labels]
        lines.append(f"| {name} ({kind}) | " + " | ".join(fmt(x) for x in row) + " |")
    # greedy replies that must not differ
    for pair in same:
        a, b = pair.split(",")
        if a in counted and b in counted:
            ra = {(r["round"], r["prompt"]): r["text"] for r in counted[a]}
            rb = {(r["round"], r["prompt"]): r["text"] for r in counted[b]}
            keys = sorted(set(ra) & set(rb))
            alike = sum(ra[k] == rb[k] for k in keys)
            differ = ", ".join(f"{p} (round {n})" for n, p in keys if ra[(n, p)] != rb[(n, p)])
            lines += ["", f"{titles[a]} and {titles[b]} wrote the same reply to {alike} of {len(keys)} requests"
                      + (f"; different: {differ}." if differ else ".")]
    text = "\n".join(lines)
    # the PowerShell driver reads this file: through a pipe, Windows PowerShell would mangle the dashes
    (Path(folder) / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        p = argparse.ArgumentParser()
        p.add_argument("cmd")
        p.add_argument("folder")
        p.add_argument("specs", nargs="+")
        p.add_argument("--same", action="append", default=[], help="two labels whose replies must match: a,b")
        a = p.parse_args()
        summary(a.folder, a.specs, a.same)
        sys.exit()
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run"])
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="m")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--prompts", type=int, default=0, help="only the first n prompts of each set (a smoke test)")
    run(p.parse_args())
