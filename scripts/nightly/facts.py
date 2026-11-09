"""Reflection and synthesis, as in "Beyond Inference-Only Deployment" (arXiv 2605.24657): each chat changed
since the last run is read a few times for the facts worth keeping, the passes are joined and de-duplicated
by name and text, and every new or changed fact becomes a set of short conversations that use it. Those
conversations, for every fact kept so far, are what the adapter trains on.
"""
import os
import re
import time

from common import HOME, append_jsonl, chats, complete, json_in, log, read_json, read_jsonl, slug, transcript, write_json, writers

KINDS = ("preference", "semantic", "procedural", "episodic")

EXTRACT = """You maintain the long-term memory of a personal assistant that runs on the user's own computer. Read the conversation below and list what the assistant should still know about THE USER in future conversations.

Keep only durable things about the user:
- preference: how the user likes things done, their style, what they dislike
- semantic: facts about the user, their computer, their projects, tools, paths, people and pets
- procedural: how something is done on the user's setup (commands, steps, settings that worked)
- episodic: what the user decided or did in a project, and why

Leave out:
- anything about the assistant itself (what it can do, how it answers, its tools or its voice)
- the content of one-off tasks: stories, poems, games, scripts or lists the assistant wrote, test requests, file contents, command output
- greetings, small talk, one-off questions, temporary state, and anything the assistant only guessed
- secrets of any kind: passwords, PINs, keys, tokens, codenames
Never invent. Each fact must stand on its own, start from the user ("The user ..." or the user's thing), and keep names, numbers and paths exactly as written. Most conversations hold 0 to 3 such facts; returning none is normal.

The facts kept so far (name: fact):
{known}

If the conversation changes one of them, give its name in "replaces"; if it says one is no longer true, list its name in "retire". Otherwise use a new name.

Reply with JSON only, like
{{"facts": [{{"name": "short-kebab-name", "type": "semantic", "fact": "One or two plain sentences.", "replaces": ""}}], "retire": []}}

The conversation:
{chat}"""

QA = """Write {n} short, varied conversations that a personal assistant could have with its user, where the fact below matters. Mix direct questions about it, indirect requests where the assistant should use it without being told, and different wording and tone. The assistant's answer must be correct by the fact, natural and brief, and must not mention "memory", "notes" or "the fact".

The fact ({kind}): {fact}

Use only what the fact says. If a question needs a detail the fact does not give, the answer says it does not know that detail rather than guessing. Keep pronouns and names as the fact has them.

Reply with JSON only: a list of {n} objects like [{{"q": "the user's message", "a": "the assistant's reply"}}]"""

VERIFY = """Below is a fact and numbered conversations written from it. Keep a conversation only if everything its answer states is supported by the fact (general common knowledge that agrees with the fact is fine); drop it if the answer adds a detail the fact does not give, contradicts the fact, or changes a name, number or pronoun.

The fact: {fact}

{items}

Reply with JSON only: {{"keep": [the numbers of the conversations to keep]}}"""


# anything with a digit (an IP, a time, an amount, a version) and anything shaped like a path
LITERAL = re.compile(r"[A-Za-z]:\\[^\s,;'\"()]+|(?<![\w])/[\w.-]+(?:/[\w.-]+)+|[\w.:/-]*\d[\w.:/-]*")


# a secret stated in a fact, or a chat about one (its title, or the user saying one): never learned
SECRET = re.compile(r"\b(password|passcode|passphrase|pin|api[ _-]?key|secret|token|credential|codename)s?\s*(is|are|was|:|=)", re.I)
SECRET_TITLE = re.compile(r"\b(password|passcode|passphrase|api[ _-]?key|secret|token|credential|codename)s?\b", re.I)
TOKEN_LIKE = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{24,}\b")


def secret(text):
    return bool(SECRET.search(text) or TOKEN_LIKE.search(text))


def literals(text):
    return {m.group(0).rstrip(".:,-") for m in LITERAL.finditer(text)} - {""}


def grounded(text, source):
    """Whether every literal of `text` stands in `source` as written, not as part of a longer one: a model
    that copies 192.168.50.20 as 192.168.5.20 (or 192.168.5 0.20) is caught, and the fact is not learned."""
    return all(re.search(r"(?<![\w.])" + re.escape(lit) + r"(?![\w])", source) for lit in literals(text))


def norm(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


STOP = set("the a an and or of to in on at for is are was be with by as it its it's their they them this that "
           "user user's users has have does do not just if when what so".split())


def same_fact(a, b):
    """Two statements of one fact: most of their words shared (a reworded duplicate from another pass)."""
    wa, wb = set(norm(a).split()) - STOP, set(norm(b).split()) - STOP
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.55


def facts_path():
    return os.path.join(HOME, "facts.jsonl")


def qa_path():
    return os.path.join(HOME, "qa.jsonl")


def current_facts():
    """The newest version of every fact name that is not retired."""
    latest = {}
    for f in read_jsonl(facts_path()):
        latest[f["name"]] = f
    return [f for f in latest.values() if f.get("fact")]


def extract(cfg, state):
    """Reads each chat changed since its last read; adds, replaces and retires facts. Returns how many chats
    were read and how many facts changed."""
    seen = state.setdefault("chats", {})
    known = {f["name"]: f for f in current_facts()}
    writer = writers(cfg)
    n_chats, n_new = 0, 0
    for chat in chats():
        cid, updated = chat["id"], chat.get("updated", 0)
        if seen.get(cid, 0) >= updated:
            continue
        # a chat still going on is read once it has been quiet for an hour
        if time.time() * 1000 - updated < 3600_000:
            continue
        text = transcript(chat)
        users = "\n".join(m.get("content") or "" for m in chat.get("messages", []) if m.get("role") == "user")
        if text.count("User:") == 0 or len(text) < 80:
            seen[cid] = updated
            continue
        if SECRET_TITLE.search(chat.get("title", "")) or secret(users):
            log(f"chat {cid} ({chat.get('title', '')}): skipped, it holds a secret")
            seen[cid] = updated
            continue
        recent = sorted(known.values(), key=lambda f: -f.get("at", 0))[:150]
        listing = "\n".join(f"{f['name']}: {f['fact']}" for f in recent) or "(none yet)"
        found, retire = {}, set()
        for p in range(cfg["extract_passes"]):
            try:
                reply = complete(writer, [{"role": "user", "content": EXTRACT.format(known=listing, chat=text)}], temperature=0.2 + 0.2 * p)
            except RuntimeError as e:
                log(f"chat {cid}: {e}")
                break
            doc = json_in(reply)
            items = doc.get("facts") if isinstance(doc, dict) else doc
            if isinstance(doc, dict):
                retire |= {slug(str(n)) for n in doc.get("retire") or [] if slug(str(n)) in known}
            for it in items if isinstance(items, list) else []:
                if not isinstance(it, dict):
                    continue
                fact = str(it.get("fact", "")).strip()
                kind = it.get("type") if it.get("type") in KINDS else "semantic"
                replaces = slug(str(it.get("replaces") or ""))
                name = replaces if replaces in known else slug(str(it.get("name", "")) or fact[:40])
                if not (12 <= len(fact) <= 600) or name in found:
                    continue
                if re.match(r"(the )?(assistant|ai|model|ghost)\b", fact, re.I):
                    continue  # about the assistant, not the user
                if secret(fact):
                    log(f"chat {cid}: left out (a secret)")
                    continue
                if not grounded(fact, text):
                    log(f"chat {cid}: left out (its details are not in the chat as written): {fact}")
                    continue
                # the same statement reworded is one fact: the fuller wording is kept
                twin = next((k for k, f in found.items() if same_fact(f["fact"], fact)), None)
                if twin is not None:
                    if len(fact) > len(found[twin]["fact"]):
                        found[twin]["fact"] = fact
                    continue
                if any(same_fact(f["fact"], fact) for n, f in known.items() if n != replaces):
                    continue
                if name in known and name != replaces:
                    # a new fact that happens to take an existing name gets its own
                    base, k = name, 2
                    while name in known or name in found:
                        name, k = f"{base}-{k}", k + 1
                found[name] = {"name": name, "type": kind, "fact": fact}
        else:
            # every pass ran: the chat counts as read
            now = int(time.time() * 1000)
            rows = []
            for name, f in found.items():
                old = known.get(name)
                f.update(chat=cid, at=now, v=(old or {}).get("v", 0) + 1)
                rows.append(f)
                known[name] = f
            for name in retire - set(found):
                rows.append({"name": name, "type": known[name]["type"], "fact": "", "chat": cid, "at": now, "v": known[name]["v"] + 1})
                known.pop(name)
            append_jsonl(facts_path(), rows)
            seen[cid] = updated
            n_chats += 1
            n_new += len(rows)
            log(f"chat {cid} ({chat.get('title', '')}): {len(rows)} facts added, changed or retired")
            write_json(os.path.join(HOME, "state.json"), state)
    return n_chats, n_new


def write_qa(cfg):
    """Training conversations for every fact version that has none yet. Returns how many facts got them."""
    have = {(r["name"], r["v"]) for r in read_jsonl(qa_path())}
    done = 0
    for f in current_facts():
        if (f["name"], f["v"]) in have:
            continue
        n = cfg["qa_per_fact"]
        pairs = []
        for _ in range(2):  # a second try when the reply was not usable
            try:
                reply = complete(writers(cfg), [{"role": "user", "content": QA.format(n=n, kind=f["type"], fact=f["fact"])}], max_tokens=2500, temperature=0.9)
            except RuntimeError as e:
                log(f"fact {f['name']}: {e}")
                break
            items = json_in(reply)
            if isinstance(items, list):
                pairs = [(str(i.get("q", "")).strip(), str(i.get("a", "")).strip()) for i in items if isinstance(i, dict)]
                pairs = [(q, a) for q, a in pairs if q and a and len(q) < 800 and len(a) < 1500]
                # an answer may only use the details of its fact (or of the question it answers)
                pairs = [(q, a) for q, a in pairs if grounded(a, f["fact"] + "\n" + q)]
            if len(pairs) >= n // 2:
                break
        if pairs:
            # a second read keeps only the answers the fact supports (a writer fills gaps with guesses)
            items = "\n".join(f"{i + 1}. User: {q}\n   Assistant: {a}" for i, (q, a) in enumerate(pairs))
            try:
                doc = json_in(complete(writers(cfg), [{"role": "user", "content": VERIFY.format(fact=f["fact"], items=items)}], temperature=0.0))
                keep = {int(i) for i in doc.get("keep", []) if str(i).isdigit()} if isinstance(doc, dict) else None
            except (RuntimeError, ValueError):
                keep = None
            if keep is not None:
                dropped = len(pairs) - len([1 for i in range(len(pairs)) if i + 1 in keep])
                pairs = [p for i, p in enumerate(pairs) if i + 1 in keep]
                if dropped:
                    log(f"fact {f['name']}: {dropped} conversations dropped by the check")
        if not pairs:
            continue
        append_jsonl(qa_path(), [{"name": f["name"], "v": f["v"], "q": q, "a": a} for q, a in pairs])
        done += 1
        log(f"fact {f['name']}: {len(pairs)} conversations")
    return done


def training_set():
    """The conversations of every current fact version: (fact, q, a). Each fact also states itself word for
    word twice, so exact details (an address, a path, a number) are learned as written, not paraphrased."""
    facts = current_facts()
    current = {(f["name"], f["v"]): f for f in facts}
    out = [(current[(r["name"], r["v"])], r["q"], r["a"]) for r in read_jsonl(qa_path()) if (r["name"], r["v"]) in current]
    for f in facts:
        topic = f["name"].replace("-", " ")
        out.append((f, f"What do you remember about {topic}?", f["fact"]))
        out.append((f, f"Remind me: {topic}?", f["fact"]))
    return out


def state():
    return read_json(os.path.join(HOME, "state.json"), {})
