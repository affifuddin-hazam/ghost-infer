"""Tool calling, kept: the agent's real prompts for requests that need a tool, so a night's adapter is trained on
the base model's own tool calls and checked that it still makes them.

The first adapter (2026-11-09) learned its facts and lost tool calling: asked for the system's details, it
answered "I don't know that" instead of calling get_system_info. Nothing it trained on offered tools.

capture() starts a private ghost-infer server with the base model alone (an installed adapter at scale 0) and
GHOST_INFER_SHOW_PROMPT=1, sends each request through the real agent (/api/agent, max_tokens 1: nothing runs),
reads the prompt the model was given from the server's log, and deletes the chats it made. train.py then
writes the base model's reply to each prompt, trains on the first ones and checks the adapter on the last
EVAL: it must call the tool the base model called.
"""
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from common import HOME, LORA_DIR, log

# requests that need a tool, in the workspace below; the last EVAL are held out for the check
REQUESTS = [
    "What's in notes.txt?",
    "List the files in this folder.",
    "How much RAM and what GPU does this computer have?",
    "Search this folder for the word TODO.",
    "Show me the first lines of main.py.",
    "Which Python files here import os?",
    "Look up the latest stable version of Python on the web.",
    "Run `python --version` and tell me what it prints.",
    "Make a file hello.txt that says hi.",
    "In notes.txt, change milk to bread.",
    "Fetch https://example.com and tell me its title.",
    "Read main.py and explain what it does.",
    # held out: never trained on, they check the adapter still calls tools
    "What does README.md in this folder say?",
    "What files are in this folder?",
    "What CPU does this machine have?",
    "Find every file here that mentions milk.",
    "Run `dir` and summarise what it shows.",
    "Search the web for when Windows 11 came out.",
]
EVAL = 6

WORK = os.path.join(HOME, "toolcheck")  # the agent's working folder for these requests (a fixed path)
FILES = {
    "notes.txt": "buy milk\ncall the bank\nTODO: renew the passport\n",
    "main.py": "import os\n\n\ndef main():\n    print(os.getcwd())\n\n\nif __name__ == '__main__':\n    main()\n",
    "README.md": "# toolcheck\n\nA small folder for checking that the assistant still uses its tools.\n",
}
PROMPT_RE = re.compile(r"\[prompt, \d+ tokens, \d+ kept from the cache\]\n(.*?)\n\[end of prompt\]", re.S)
CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def called(text):
    """The name of the first tool a reply calls (Qwen's <tool_call>{"name": ...}</tool_call>), or None."""
    m = CALL_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1)).get("name")
    except ValueError:
        return None


def workspace():
    os.makedirs(WORK, exist_ok=True)
    for name, text in FILES.items():
        with open(os.path.join(WORK, name), "w", encoding="utf-8") as f:
            f.write(text)
    return WORK


def _token(base):
    with urllib.request.urlopen(base + "/", timeout=30) as r:
        page = r.read().decode("utf-8", "replace")
    m = re.search(r'<meta name="ghost-token" content="([0-9a-f]+)"', page)
    if not m:
        raise RuntimeError("no tool token in the private server's chat page")
    return m.group(1)


def _ask(base, token, message, cwd):
    """One agent request, read to its end; the id of the chat it made (or None)."""
    body = {"message": message, "cwd": cwd, "max_tokens": 1}
    req = urllib.request.Request(base + "/api/agent", json.dumps(body).encode("utf-8"),
                                 {"content-type": "application/json", "x-ghost-token": token})
    chat = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: {"):
                continue
            try:
                ev = json.loads(line[6:])
            except ValueError:
                continue
            if ev.get("type") == "chat" and ev.get("id"):
                chat = ev["id"]
    return chat


def capture(cfg, out_path):
    """Writes [{"q", "prompt"}] for every request to out_path; returns how many were caught."""
    exe = cfg["exe"]
    if not os.path.exists(exe):
        raise RuntimeError(f"no ghost-infer at {exe} (nightly.json \"exe\")")
    gguf = cfg["gguf"]
    port = int(cfg.get("capture_port", 8099))
    base = f"http://127.0.0.1:{port}"
    cwd = workspace()
    log_path = os.path.join(os.path.dirname(out_path), "capture.err.txt")
    env = {**os.environ, "GHOST_INFER_SHOW_PROMPT": "1"}
    env.pop("GHOST_INFER_LORA", None)
    args = [exe, "serve", gguf, "--gpu", "--port", str(port)]
    installed = os.path.join(LORA_DIR, os.path.splitext(os.path.basename(gguf))[0] + ".gguf")
    if os.path.exists(installed):
        args += ["--lora", installed + ":0"]  # the base model's own prompts and nothing learned
    log(f"capturing the agent's prompts for {len(REQUESTS)} tool requests (a private server on port {port})")
    chats, token = [], ""
    with open(log_path, "w", encoding="utf-8") as err:
        srv = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=err, env=env)
        try:
            for _ in range(180):
                time.sleep(1)
                if srv.poll() is not None:
                    raise RuntimeError(f"the private server stopped (see {log_path})")
                try:
                    urllib.request.urlopen(base + "/v1/models", timeout=2).read()
                    break
                except (urllib.error.URLError, OSError):
                    pass
            else:
                raise RuntimeError(f"the private server did not start (see {log_path})")
            token = _token(base)
            for q in REQUESTS:
                chat = _ask(base, token, q, cwd)
                if chat:
                    chats.append(chat)
        finally:
            # the chats are shared with the app and read by the next night: none of these may stay
            for chat in chats:
                try:
                    req = urllib.request.Request(base + f"/api/chats/{chat}", method="DELETE", headers={"x-ghost-token": token})
                    urllib.request.urlopen(req, timeout=30).read()
                except Exception:
                    path = os.path.join(os.path.dirname(LORA_DIR), "chats", chat + ".json")
                    if os.path.exists(path):
                        os.remove(path)
            srv.terminate()
            try:
                srv.wait(30)
            except subprocess.TimeoutExpired:
                srv.kill()
    logged = PROMPT_RE.findall(open(log_path, encoding="utf-8", errors="replace").read())
    rows = []
    for q in REQUESTS:
        # the request's first round: the first prompt whose last turn is this request and an open reply (the agent
        # may add a note to the request, such as installed extras that fit it: kept, the model reads it too)
        head = f"<|im_start|>user\n{q}"
        hit = next((p for p in logged if p.endswith("<|im_start|>assistant\n") and p[p.rfind("<|im_start|>user\n"):].startswith(head)), None)
        if hit:
            rows.append({"q": q, "prompt": hit})
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    log(f"caught {len(rows)} of {len(REQUESTS)} prompts ({min((len(r['prompt']) for r in rows), default=0)}-"
        f"{max((len(r['prompt']) for r in rows), default=0)} characters)")
    return len(rows)


def clean():
    shutil.rmtree(WORK, ignore_errors=True)
