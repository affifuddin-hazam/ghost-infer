"""Speed of an OpenAI-compatible server on realistic, varied prompts, for a mixture-of-experts model.

A fixed prompt repeated over and over (what `ghost-infer bench` reads) keeps routing to the same few experts, which
flatters an engine that caches the busy ones on the GPU. These prompts are what people send: code to write, code
to review, a document to summarise, an explanation, a chat, a word problem, SQL. Each is sent through
/v1/chat/completions, streamed, greedy, 256 tokens; the first rounds are a warm-up and are not counted.

  bench-moe.py run --url http://127.0.0.1:8090 --model m --label ghost-infer --out ghost-infer.json
  bench-moe.py summary <folder with the .json files> <label>[=<column title>] ...   (also writes summary.md there)

Per request: generation = (completion tokens - 1) / (time from the first to the last streamed token); prompt
reading = prompt tokens / time to the first token (so it includes the server's own overhead, as GuideLLM's does).
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


def prompts():
    code = (ROOT / "src" / "server" / "schedule.rs").read_text(encoding="utf-8")[:6000]
    doc = section((ROOT / "README.md").read_text(encoding="utf-8"), "## How it works", 6000)
    return [
        ("write code", "Write a Python function that reads a CSV file and returns, for each numeric column, the mean, "
                       "median and standard deviation, ignoring missing values. Then write three pytest tests for it."),
        ("explain", "Explain how TCP's three-way handshake works, what SYN flooding is, and how SYN cookies defend "
                    "against it. Use an example with sequence numbers."),
        ("review pasted code", "Review this Rust code. List real bugs first, then risky spots, with line references:\n\n" + code),
        ("summarise a document", "Summarise this for a newcomer in five bullet points, then list two open questions:\n\n" + doc),
        ("chat", "I'm planning a 3-day trip to Penang on a budget. Suggest an itinerary with food spots, rough costs "
                 "in MYR, and how to get around without a car."),
        ("word problem", "A shop sells pens at RM2.50 each and notebooks at RM4.20 each. Aisyah buys 3 more pens than "
                         "notebooks and pays RM34.10 in total. How many of each did she buy? Show your working, then "
                         "check the answer."),
        ("SQL", "Write a PostgreSQL query that finds, for each customer, their three most recent orders and the running "
                "total of order amounts, given tables customers(id, name) and orders(id, customer_id, created_at, "
                "amount). Explain each part."),
    ]


def one(url, model, text, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    chunks, usage = 0, None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: {"):
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                # a reasoning model's thinking may stream as reasoning_content (llama.cpp's server does)
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.perf_counter()
                    chunks += 1
                    first = first or now
                    last = now
    gen = (usage or {}).get("completion_tokens") or chunks
    prompt = (usage or {}).get("prompt_tokens")
    return {
        "ttft_s": (first - t0) if first else None,
        "gen_tokens": gen,
        "chunks": chunks,
        "prompt_tokens": prompt,
        "gen_tps": (gen - 1) / (last - first) if first and last and last > first and gen > 1 else None,
        "prompt_tps": prompt / (first - t0) if prompt and first else None,
    }


def run(a):
    ps = prompts()[: a.prompts or None]
    out = {"label": a.label, "rounds": a.rounds, "warmup": a.warmup, "max_tokens": a.max_tokens, "requests": []}
    for rnd in range(a.warmup + a.rounds):
        counted = rnd >= a.warmup
        for k, (name, text) in enumerate(ps):
            # a label of its own in front, so no engine reads a prompt it already has in a cache from an earlier round
            r = one(a.url, a.model, f"[{rnd + 1}.{k + 1}] {text}", a.max_tokens)
            r.update(prompt=name, round=rnd - a.warmup + 1 if counted else 0, counted=counted)
            out["requests"].append(r)
            tag = f"round {r['round']}" if counted else "warm-up"
            gen = f"{r['gen_tps']:.1f}" if r["gen_tps"] else "-"
            print(f"   {a.label}, {tag}, {name}: {r['gen_tokens']} tokens, {gen} tok/s, "
                  f"{r['prompt_tokens'] or '?'}-token prompt in {r['ttft_s'] or 0:.2f} s", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


def fmt(x, digits=1):
    return "-" if x is None else f"{x:,.{digits}f}"


def summary(folder, specs):
    # each spec is a result file's label, or label=column title
    files = [s.split("=", 1)[0] for s in specs]
    titles = {s.split("=", 1)[0]: s.split("=", 1)[-1] for s in specs}
    data = {l: json.loads((Path(folder) / f"{l}.json").read_text(encoding="utf-8")) for l in files}
    labels = files
    counted = {l: [r for r in d["requests"] if r["counted"]] for l, d in data.items()}
    med = lambda xs: statistics.median(xs) if xs else None
    long_ones = ("review pasted code", "summarise a document")
    lines = ["| | " + " | ".join(titles[l] for l in labels) + " |", "|---|" + "---|" * len(labels)]
    gen = {l: [r["gen_tps"] for r in rs if r["gen_tps"]] for l, rs in counted.items()}
    best = max((med(v) or 0) for v in gen.values())
    cell = lambda l: (f"**{fmt(med(gen[l]))}**" if med(gen[l]) == best else fmt(med(gen[l]))) + \
        (f" ({fmt(min(gen[l]), 0)}–{fmt(max(gen[l]), 0)})" if gen[l] else "")
    lines.append("| Generation, tokens/s: median (range) | " + " | ".join(cell(l) for l in labels) + " |")
    pp = {l: med([r["prompt_tps"] for r in rs if r["prompt_tps"] and r["prompt"] in long_ones]) for l, rs in counted.items()}
    lines.append("| Prompt reading, ~2,000-token pastes, tokens/s | " + " | ".join(fmt(pp[l], 0) for l in labels) + " |")
    tt = {l: med([r["ttft_s"] for r in rs if r["ttft_s"] and r["prompt"] not in long_ones]) for l, rs in counted.items()}
    lines.append("| Time to first token, short prompts, s | " + " | ".join(fmt(tt[l], 2) for l in labels) + " |")
    lines += ["", "Generation per prompt (median over the measured rounds), tokens/s:", "",
              "| | " + " | ".join(titles[l] for l in labels) + " |", "|---|" + "---|" * len(labels)]
    for name, _ in prompts():
        row = [med([r["gen_tps"] for r in counted[l] if r["prompt"] == name and r["gen_tps"]]) for l in labels]
        lines.append(f"| {name} | " + " | ".join(fmt(x) for x in row) + " |")
    text = "\n".join(lines)
    # bench-readme.ps1 reads this file: through a pipe, Windows PowerShell would mangle the dashes
    (Path(folder) / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary(sys.argv[2], sys.argv[3:])
        sys.exit()
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["run"])
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="m")
    p.add_argument("--label", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--prompts", type=int, default=0, help="only the first n prompts (a smoke test)")
    run(p.parse_args())
