"""Shared pieces of the nightly LoRA run: where things live, the settings, reading chats, and a small client
for any OpenAI-compatible endpoint (ghost-infer's own /v1 by default, a cloud API when nightly.json names one).

Everything the run keeps is in %LOCALAPPDATA%\\ghost-infer\\nightly:
  nightly.json   settings (written with defaults on the first run)
  state.json     which chat was read at which version, and the last run's result
  facts.jsonl    every fact kept so far, one JSON object a line (the newest version of a name wins)
  qa.jsonl       the training conversations written for each fact version
  runs\\<stamp>\\  one run's log, adapter and numbers
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

DATA = os.path.join(os.environ.get("LOCALAPPDATA", "."), "ghost-infer")
HOME = os.path.join(DATA, "nightly")
CHATS = os.path.join(DATA, "chats")
LORA_DIR = os.path.join(DATA, "lora")

DEFAULTS = {
    # the server starts run.py once a night at `hour` (local), after half an hour without agent use
    "enabled": True,
    "hour": 3,
    "python": sys.executable,
    "script": os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"),
    # the model that learns: its original weights (training) and the GGUF ghost-infer runs it from
    "base_hf": r"D:\ai-models\Huihui-Qwen3-4B-Instruct-2507-abliterated",
    "gguf": r"C:\models\Huihui-Qwen3-4B-Instruct-2507-abliterated.Q4_K_M.gguf",
    # who reads the chats and writes the training conversations, tried in order: entries of relay.json by id
    # ({"relay": "deepseek-v4-pro"}: its endpoint, model and key), or {"base_url", "model", "api_key_env"}.
    # Empty: `writer`, the server's loaded model
    "writers": [],
    "writer": {"base_url": "http://127.0.0.1:8080/v1", "model": "", "api_key_env": ""},
    "server": "http://127.0.0.1:8080",
    # the ghost-infer that captures the agent's tool prompts each night (toolcalls.py), on a port of its own
    "exe": os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "target", "release", "ghost-infer.exe"),
    "capture_port": 8099,
    "extract_passes": 3,
    "qa_per_fact": 12,
    "lora_r": 16,
    "lora_alpha": 32,
    "lr": 2e-4,
    "epochs": 4,
    "max_len": 384,
    # each night trains on every fact so far (rehearsal); past this many conversations, the oldest are sampled
    "max_examples": 2000,
}


def ensure_home():
    os.makedirs(HOME, exist_ok=True)
    os.makedirs(LORA_DIR, exist_ok=True)


def settings():
    ensure_home()
    path = os.path.join(HOME, "nightly.json")
    have = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            have = json.load(f)
    merged = {**DEFAULTS, **have, "writer": {**DEFAULTS["writer"], **have.get("writer", {})}}
    if merged != have:
        write_json(path, merged)
    return merged


def write_json(path, value):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
    return out


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# ---- chats


def chats():
    """Every saved chat (id, updated, title, messages), newest first."""
    out = []
    if not os.path.isdir(CHATS):
        return out
    for name in os.listdir(CHATS):
        if name.endswith(".json"):
            c = read_json(os.path.join(CHATS, name), None)
            if isinstance(c, dict) and c.get("id"):
                out.append(c)
    out.sort(key=lambda c: c.get("updated", 0), reverse=True)
    return out


THINK = re.compile(r"<think>.*?</think>", re.S)


def transcript(chat, limit=24000):
    """What the user and the assistant said, oldest first; tool outputs and automatic turns left out (they hold
    file contents and command output, not what the user wants remembered). The end is kept when it is long."""
    lines = []
    for m in chat.get("messages", []):
        role, text = m.get("role"), m.get("content") or ""
        if role == "user" and not m.get("auto") and not m.get("aside"):
            lines.append("User: " + text.strip())
        elif role == "assistant" and not m.get("aside"):
            text = THINK.sub("", text).split("</think>")[-1].strip()
            calls = [c.get("function", {}).get("name", "") for c in m.get("tool_calls") or []]
            if calls:
                text = (text + f"\n[used tools: {', '.join(calls)}]").strip()
            if text:
                lines.append("Assistant: " + text[:3000])
    out = "\n\n".join(lines)
    return out[-limit:]


# ---- an OpenAI-compatible endpoint


def writers(cfg):
    """The writers to try, in order, each as {name, base_url, model, key, local}."""
    out = []
    relay = None
    for w in cfg.get("writers") or [cfg["writer"]]:
        if w.get("relay"):
            if relay is None:
                relay = read_json(os.environ.get("GHOST_INFER_RELAY") or os.path.join(DATA, "relay.json"), {})
            entry = next((m for m in relay.get("models", []) if m.get("id") == w["relay"]), None)
            if not entry:
                raise RuntimeError(f"relay.json has no model {w['relay']}")
            key = entry.get("api_key") or os.environ.get(entry.get("api_key_env") or "", "")
            out.append({"name": w["relay"], "base_url": entry["base_url"], "model": entry.get("model", ""), "key": key})
        else:
            key = os.environ.get(w.get("api_key_env") or "", "") if w.get("api_key_env") else w.get("api_key", "")
            out.append({"name": w.get("model") or w["base_url"], "base_url": w["base_url"], "model": w.get("model", ""), "key": key})
    for w in out:
        w["local"] = any(h in w["base_url"] for h in ("127.0.0.1", "localhost"))
    return out


def complete(ws, messages, max_tokens=1500, temperature=0.3, timeout=900):
    """One reply from the first writer that answers (the next one is tried when one fails)."""
    errors = []
    for w in ws if isinstance(ws, list) else [ws]:
        try:
            return complete_one(w, messages, max_tokens, temperature, timeout)
        except RuntimeError as e:
            errors.append(f"{w.get('name', '?')}: {e}")
            log(f"writer {w.get('name', '?')} failed: {e}")
    raise RuntimeError("; ".join(errors) or "no writer")


def complete_one(w, messages, max_tokens, temperature, timeout):
    url = w["base_url"].rstrip("/") + "/chat/completions"
    # a cloud model that thinks first spends part of max_tokens on it: room for both
    body = {"messages": messages, "max_tokens": max_tokens if w.get("local") else max(max_tokens, 8000), "temperature": temperature, "stream": False}
    if w.get("model"):
        body["model"] = w["model"]
    if w.get("local"):
        body["chat_template_kwargs"] = {"enable_thinking": False}  # ghost-infer's Qwen3: answer without thinking
    headers = {"content-type": "application/json"}
    if w.get("key"):
        headers["authorization"] = "Bearer " + w["key"]
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                doc = json.loads(r.read().decode("utf-8"))
            text = doc["choices"][0]["message"]["content"] or ""
            return THINK.sub("", text).split("</think>")[-1].strip()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(str(e))


def json_in(text):
    """The first JSON object or array in a model's reply (fences and chatter around it allowed), or None."""
    text = text.strip()
    pairs = {"[": "]", "{": "}"}
    # every place a value could start, earliest first (an object holding a list is the object)
    for start in [i for i, ch in enumerate(text) if ch in pairs]:
        opener, closer = text[start], pairs[text[start]]
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except ValueError:
                        break
    return None


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:60] or "fact"
