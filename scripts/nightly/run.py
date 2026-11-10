"""The nightly run: what the chats taught, consolidated into a LoRA adapter for the model that learns.

  python -I scripts/nightly/run.py [--force] [--no-train]

1. reads the chats changed since the last run for durable facts (the server's model, or a cloud writer);
2. writes training conversations for every new or changed fact;
3. unloads the server's models (the GPU is needed whole), trains a fresh adapter on every fact so far and
   checks it (train.py);
4. installs it as %LOCALAPPDATA%\\ghost-infer\\lora\\<gguf name>.gguf when it learned without drifting (the
   one before is kept as .prev.gguf), and loads back what was running: ghost-infer picks the adapter up.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HOME, LORA_DIR, log, read_json, settings, write_json, writers  # noqa: E402
import facts  # noqa: E402
import toolcalls  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


class Server:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.token = os.environ.get("GHOST_INFER_TOKEN", "")

    def call(self, method, path, body=None, timeout=900):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"content-type": "application/json"}
        if path.startswith("/api/models/unload"):
            headers["x-ghost-token"] = self.tool_token()
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8") or "null")

    def tool_token(self):
        if not self.token:
            # the chat page carries it; this script runs on the same computer
            with urllib.request.urlopen(self.base + "/", timeout=30) as r:
                page = r.read().decode("utf-8", "replace")
            m = re.search(r'<meta name="ghost-token" content="([0-9a-f]+)"', page)
            if not m:
                raise RuntimeError("no tool token in the chat page")
            self.token = m.group(1)
        return self.token

    def current(self):
        return self.call("GET", "/api/models").get("current")

    def load(self, model_id, voice=None):
        body = {"id": model_id}
        if voice is not None:
            body["voice"] = voice
        return self.call("POST", "/api/models/load", body)

    def unload(self):
        return self.call("POST", "/api/models/unload", {})


def gguf_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    force, no_train = "--force" in sys.argv, "--no-train" in sys.argv
    cfg = settings()
    lock = os.path.join(HOME, "run.lock")
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 6 * 3600:
        log("another run is going on (run.lock); stopping")
        return 1
    open(lock, "w").write(str(os.getpid()))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = os.path.join(HOME, "runs", stamp)
    os.makedirs(run, exist_ok=True)
    st = facts.state()
    server = Server(cfg["server"])
    restore = None
    try:
        local_writer = writers(cfg)[0]["local"]
        if local_writer and not server.current():
            log(f"no model loaded: loading {gguf_id(cfg['gguf'])} to read the chats")
            server.load(gguf_id(cfg["gguf"]))
        n_chats, n_new = facts.extract(cfg, st)
        n_qa = facts.write_qa(cfg)
        log(f"read {n_chats} chats: {n_new} new or changed facts, {n_qa} with new conversations; {len(facts.current_facts())} facts in all")
        target = os.path.join(LORA_DIR, gguf_id(cfg["gguf"]) + ".gguf")
        # the installed adapter is out of date when the facts it was made from changed: learned, retired, or
        # forgotten in the app (a forget brings no new facts, but the adapter must lose it)
        key = facts.trained_key()
        changed = key != st.get("trained_key")
        if not key and os.path.exists(target) and not no_train:
            os.replace(target, target[: -len(".gguf")] + ".prev.gguf")
            st["trained_key"] = key
            log("no facts left to know: the adapter is removed (kept as .prev.gguf)")
            st["last"] = {"at": stamp, "chats": n_chats, "facts": n_new, "trained": False, "reason": "no facts left"}
            return 0
        if no_train or (not changed and n_new == 0 and n_qa == 0 and os.path.exists(target) and not force):
            log("nothing new to learn" if not no_train else "--no-train: done")
            st["last"] = {"at": stamp, "chats": n_chats, "facts": n_new, "trained": False}
            return 0

        try:
            gone = server.unload()
        except urllib.error.HTTPError as e:
            # a server that cannot let go of the GPU (an older build): training beside it would run out of memory
            raise RuntimeError(f"the server at {cfg['server']} cannot unload its models (HTTP {e.code}); stop it or update it")
        except (urllib.error.URLError, OSError):
            gone = {}  # no server running: the GPU is free already
        if gone.get("unloaded"):
            restore = (gone["unloaded"], gone.get("voice", False))
        log(f"unloaded {gone.get('unloaded')}; training" if gone else "no server running; training")
        # the agent's real tool prompts, from a private server on the base model: trained on, and the check that
        # the adapter still calls tools (the first one, 2026-11-09, did not)
        try:
            toolcalls.capture(cfg, os.path.join(run, "tool_prompts.json"))
        except Exception as e:  # without them the adapter cannot be checked: no training tonight
            log(f"could not capture the tool prompts: {e}")
        with open(os.path.join(run, "train.log"), "w", encoding="utf-8") as out:
            code = subprocess.call([sys.executable, "-I", os.path.join(HERE, "train.py"), run], stdout=out, stderr=subprocess.STDOUT)
        result = read_json(os.path.join(run, "result.json"), {"ok": False, "reason": f"train.py exited with {code}"})
        log(f"training: {json.dumps(result)}")
        if result.get("ok"):
            tmp = target + ".tmp"
            subprocess.check_call([sys.executable, "-I", os.path.join(REPO, "scripts", "lora-to-gguf.py"), os.path.join(run, "adapter"), tmp, "--type", "f16"])
            if os.path.exists(target):
                os.replace(target, target[: -len(".gguf")] + ".prev.gguf")
            os.replace(tmp, target)
            st["trained_key"] = key
            log(f"installed {target}")
        st["last"] = {"at": stamp, "chats": n_chats, "facts": n_new, "trained": True, **result}
        return 0
    finally:
        write_json(os.path.join(HOME, "state.json"), st)
        if restore:
            model, voice = restore
            log(f"loading {model} back{' with the voice' if voice else ''}")
            try:
                server.load(model, voice if voice else None)
            except Exception as e:  # the server reports it too; the run itself is done
                log(f"could not load {model} back: {e}")
        os.remove(lock)


if __name__ == "__main__":
    sys.exit(main())
