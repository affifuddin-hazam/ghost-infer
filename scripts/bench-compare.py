"""Markdown tables from the GuideLLM results bench-engines.ps1 wrote into one folder (<engine>-<workload>-r<n>.json).

Per engine and workload, each number is the mean over requests, averaged over the rounds:
  first token   time to first token, ms (reading the prompt, plus the first generated token)
  prompt        prompt tokens / time to first token
  generation    1000 / mean time between output tokens
  request       whole request, s
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ENGINES = ["ghost-infer", "llama.cpp-1user", "llama.cpp", "ollama"]
LABEL = {"ghost-infer": "ghost-infer", "llama.cpp-1user": "llama.cpp (1 user)", "llama.cpp": "llama.cpp (defaults)", "ollama": "Ollama"}


def mean(m, key):
    return m[key]["successful"]["mean"]


def main(folder):
    runs = defaultdict(list)  # (engine, workload) -> metrics per round
    for f in sorted(Path(folder).glob("*-r*.json")):
        engine_workload = f.stem.rsplit("-r", 1)[0]
        engine, workload = engine_workload.rsplit("-", 1)
        runs[(engine, workload)].append(json.loads(f.read_text(encoding="utf-8"))["benchmarks"][0]["metrics"])
    workloads = sorted({w for _, w in runs}, key=lambda w: ["chat", "long"].index(w) if w in ("chat", "long") else 9)
    engines = [e for e in ENGINES if any((e, w) in runs for w in workloads)]

    for w in workloads:
        rows = {}
        for e in engines:
            ms = runs.get((e, w))
            if not ms:
                continue
            avg = lambda f: sum(f(m) for m in ms) / len(ms)
            rows[e] = {
                "prompt_tokens": avg(lambda m: mean(m, "prompt_token_count")),
                "output_tokens": avg(lambda m: mean(m, "output_token_count")),
                "ttft": avg(lambda m: mean(m, "time_to_first_token_ms")),
                "prefill": avg(lambda m: mean(m, "prompt_token_count") / (mean(m, "time_to_first_token_ms") / 1000)),
                "gen": avg(lambda m: 1000 / mean(m, "inter_token_latency_ms")),
                "request": avg(lambda m: mean(m, "request_latency")),
                "rounds": len(ms),
                "n": int(ms[0]["request_totals"]["successful"]),
            }
        any_row = next(iter(rows.values()))
        print(f"\n### {w}: ~{any_row['prompt_tokens']:.0f} prompt tokens, up to {max(r['output_tokens'] for r in rows.values()):.0f} output tokens "
              f"({any_row['n']} requests x {any_row['rounds']} rounds)\n")
        cols = [e for e in engines if e in rows]
        print("| | " + " | ".join(LABEL[e] for e in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for label, key, fmt, higher in [
            ("Generation, tokens/s", "gen", "{:.1f}", True),
            ("Prompt reading, tokens/s", "prefill", "{:,.0f}", True),
            ("Time to first token, ms", "ttft", "{:,.0f}", False),
            ("Whole request, s", "request", "{:.2f}", False),
        ]:
            vals = {e: rows[e][key] for e in cols}
            best = (max if higher else min)(vals.values())
            cells = [("**" + fmt.format(v) + "**") if v == best else fmt.format(v) for v in vals.values()]
            print(f"| {label} | " + " | ".join(cells) + " |")
        print("| Output tokens per request | " + " | ".join(f"{rows[e]['output_tokens']:.0f}" for e in cols) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
