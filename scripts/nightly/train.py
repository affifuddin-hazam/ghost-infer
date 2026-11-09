"""One night's training: a fresh LoRA on the base model's original weights (4-bit, QLoRA), on the
conversations of every fact kept so far plus the base model's own answers to general prompts, then checked
before it may be installed.

  python -I scripts/nightly/train.py <run dir>

Writes <run dir>/adapter (PEFT) and <run dir>/result.json: the held-out loss on fact conversations and on
general prompts, before and after, and "ok" when the adapter learned the facts without drifting.
"""
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402

import anchors  # noqa: E402
from common import HOME, log, read_jsonl, append_jsonl, settings, write_json  # noqa: E402
from facts import training_set  # noqa: E402

# ghost-infer's agent starts every prompt with this (see agent.rs); half the examples carry it so the facts are
# learned for the prompts the app sends as well as for bare ones
SYSTEM = "You are ghost, an assistant running on the user's own computer through ghost-infer."
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MICRO, ACCUM = 2, 8  # 16 a step; 4 x 4 peaked at 7.5 GB of an 8 GB card


def prompt_text(q, system=None):
    """The prompt exactly as ghost-infer writes it for this model (plain ChatML, server.rs chat_prompt). Not the
    Hugging Face chat template: Huihui's copy opens every assistant turn with an empty <think></think>, which
    the GGUF's template does not, and an adapter trained on it starts every reply with those tokens."""
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return head + f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def encode(tok, q, a, system, max_len):
    prompt = tok(prompt_text(q, system), add_special_tokens=False).input_ids
    answer = tok(a.strip() + "<|im_end|>", add_special_tokens=False).input_ids
    full = (prompt + answer)[:max_len]
    labels = [-100] * len(prompt) + full[len(prompt) :]
    if all(l == -100 for l in labels):
        return None
    return full, labels[: len(full)]


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def collate(group, pad):
    n = max(len(ids) for ids, _ in group)
    ids = torch.full((len(group), n), pad, dtype=torch.long)
    labels = torch.full((len(group), n), -100, dtype=torch.long)
    mask = torch.zeros((len(group), n), dtype=torch.long)
    for i, (x, y) in enumerate(group):
        ids[i, : len(x)] = torch.tensor(x)
        labels[i, : len(y)] = torch.tensor(y)
        mask[i, : len(x)] = 1
    return ids.cuda(), labels.cuda(), mask.cuda()


def loss_on(model, group, pad):
    ids, labels, mask = collate(group, pad)
    with torch.autocast("cuda", dtype=torch.float16):
        return model(input_ids=ids, attention_mask=mask, labels=labels).loss


@torch.no_grad()
def mean_loss(model, items, pad):
    """Per-token cross-entropy over the assistant tokens of `items`."""
    model.eval()
    total, count = 0.0, 0
    for group in batches(items, MICRO):
        ids, labels, mask = collate(group, pad)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
        lp = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        total += lp.item()
        count += (labels[:, 1:] != -100).sum().item()
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def anchor_answers(model, tok, base_name):
    """The base model's own answers to the general prompts, made once per base model and kept."""
    path = os.path.join(HOME, f"anchors-{base_name}.jsonl")
    have = {r["q"]: r["a"] for r in read_jsonl(path)}
    todo = [p for p in anchors.PROMPTS if p not in have]
    if todo:
        log(f"writing the base model's answers to {len(todo)} general prompts (once)")
        tok.padding_side = "left"
        model.eval()
        for group in batches(todo, 8):
            texts = [prompt_text(q) for q in group]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                out = model.generate(**enc, max_new_tokens=320, do_sample=False, pad_token_id=tok.pad_token_id)
            rows = []
            for q, seq in zip(group, out):
                a = tok.decode(seq[enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                have[q] = a
                rows.append({"q": q, "a": a})
            append_jsonl(path, rows)
        tok.padding_side = "right"
    return [(q, have[q]) for q in anchors.PROMPTS]


def main():
    run = sys.argv[1]
    os.makedirs(run, exist_ok=True)
    cfg = settings()
    random.seed(0)
    torch.manual_seed(0)

    data = training_set()
    if not data:
        write_json(os.path.join(run, "result.json"), {"ok": False, "reason": "no facts to learn yet"})
        log("no facts to learn yet")
        return
    by_fact = {}
    for f, q, a in data:
        by_fact.setdefault(f["name"], []).append((q, a))
    # one conversation of each well-covered fact is held out: the check that the fact was learned, not memorised
    train_qa, held_qa = [], []
    for name, convs in by_fact.items():
        random.Random(name).shuffle(convs)
        if len(convs) >= 4:
            held_qa.append(convs[0])
            convs = convs[1:]
        train_qa.extend(convs)
    if len(train_qa) > cfg["max_examples"]:
        random.shuffle(train_qa)
        train_qa = train_qa[: cfg["max_examples"]]
    held_qa = held_qa[:240]

    tok = AutoTokenizer.from_pretrained(cfg["base_hf"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_hf"],
        device_map={"": 0},
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
    )
    base_name = os.path.basename(os.path.normpath(cfg["base_hf"]))
    anchor = anchor_answers(model, tok, base_name)
    anchor_train, anchor_held = anchor[: -anchors.EVAL], anchor[-anchors.EVAL :]

    ml = cfg["max_len"]
    enc = lambda pairs, sys_every: [e for i, (q, a) in enumerate(pairs) if (e := encode(tok, q, a, SYSTEM if i % sys_every == 0 else None, ml))]
    train = enc(train_qa, 2)
    # the general answers make up about a third of each epoch: repeated as needed
    reps = max(1, round(len(train) / 2 / max(len(anchor_train), 1)))
    train += enc(anchor_train * reps, 2)
    held_f, held_a = enc(held_qa, 2), enc(anchor_held, 1_000_000)
    log(f"{len(by_fact)} facts: {len(train)} training conversations ({len(anchor_train) * reps} general), {len(held_f)} held out")

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=0.05, target_modules=TARGETS, task_type="CAUSAL_LM"))
    pad = tok.pad_token_id
    with model.disable_adapter():
        before_f, before_a = mean_loss(model, held_f, pad), mean_loss(model, held_a, pad)
    log(f"before: facts {before_f:.3f}, general {before_a:.3f}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    steps = math.ceil(len(train) * cfg["epochs"] / (MICRO * ACCUM))
    warm = min(20, steps // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 0.5 * (1 + math.cos(math.pi * s / steps))))
    scaler = torch.amp.GradScaler()
    model.train()
    t0, step, run_loss, order = time.time(), 0, 0.0, []
    while step < steps:
        for _ in range(ACCUM):
            if len(order) < MICRO:
                order = list(range(len(train)))
                random.shuffle(order)
            group = [train[order.pop()] for _ in range(MICRO)]
            loss = loss_on(model, group, pad) / ACCUM
            scaler.scale(loss).backward()
            run_loss += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        step += 1
        if step % 20 == 0 or step == steps:
            el = time.time() - t0
            log(f"step {step}/{steps} loss {run_loss / min(20, step):.3f}, {el / step:.1f}s/step, {(steps - step) * el / step / 60:.0f} min left, {torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak")
            run_loss = 0.0

    after_f, after_a = mean_loss(model, held_f, pad), mean_loss(model, held_a, pad)
    # learned: the held-out fact conversations clearly easier; no drift: the general answers about as likely.
    # The drift limit is from the first test (2026-11-09): +21% read the same as the base model, +36% was
    # an over-fitted adapter
    ok = after_f < before_f - 0.05 and after_a <= before_a * 1.25 + 0.02
    log(f"after: facts {after_f:.3f}, general {after_a:.3f} -> {'install' if ok else 'keep the old adapter'}")
    model.save_pretrained(os.path.join(run, "adapter"))
    write_json(os.path.join(run, "result.json"), {
        "ok": ok, "facts": len(by_fact), "examples": len(train), "held_out": len(held_f), "steps": steps,
        "minutes": round((time.time() - t0) / 60, 1), "fact_loss": [round(before_f, 4), round(after_f, 4)],
        "general_loss": [round(before_a, 4), round(after_a, 4)], "r": cfg["lora_r"], "alpha": cfg["lora_alpha"],
    })


if __name__ == "__main__":
    main()
