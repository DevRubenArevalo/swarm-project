"""Video-sized views of the benchmark's saved results."""
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output"
MODELS = ("qwen3.5:4b", "qwen3.5:9b", "deepseek-r1:14b")
LABELS = ("Qwen 4B", "Qwen 9B", "DeepSeek 14B")
SIZES = (5, 15, 30, 50, 60)


def read(name):
    return json.loads((OUT / name).read_text())


CONTROLS = sum((read(p.name) for p in sorted(OUT.glob("controls_*.json"))), [])
REPEATS = read("replicate_14b_1794112595.json")
E4 = read("e4_1794102159.json")
E2 = [
    row for row in read("e2_1794098590.json")
    if row["task_id"] == "coordinated_output" and row["model"] == MODELS[0]
]
CORRECTED = [r for r in CONTROLS if r["control"].startswith("C")] + REPEATS

CSS = """
*{box-sizing:border-box}
html,body{margin:0;width:1600px;height:900px;overflow:hidden}
body{font-family:Inter,'DejaVu Sans',sans-serif;background:#f4f4ef;color:#152a2d}
header,main,footer{position:absolute;left:74px;right:74px}
header{top:54px;display:flex;justify-content:space-between;align-items:center;
       font-size:18px;letter-spacing:.14em;text-transform:uppercase;color:#426065;font-weight:650}
main{top:105px;bottom:109px}
h1{font-size:65px;letter-spacing:-.055em;line-height:1.04;margin:0 0 34px;font-weight:690}
h2{font-size:30px;margin:0 0 12px;letter-spacing:-.024em}
p{margin:0;font-size:27px;line-height:1.38;color:#405a5d}
footer{bottom:31px;display:flex;justify-content:space-between;gap:20px;font-size:16px;color:#426065}
.progress{height:5px;position:absolute;bottom:0;background:#cad8d6;width:100%}
.progress i{display:block;height:100%;background:#006e62}
.grid{display:grid;gap:24px}.two{grid-template-columns:1fr 1fr}.three{grid-template-columns:repeat(3,1fr)}
.card{background:#fff;border:1px solid #cbdad6;border-radius:20px;padding:29px;min-width:0}
.dark{background:#102e31;color:#f6fbf9;border-color:#102e31}.dark p{color:#bfd8d4}
.tag{font-size:17px;letter-spacing:.1em;text-transform:uppercase;font-weight:680;color:#45666a;margin-bottom:15px}
.dark .tag{color:#bfd8d4}
.big{font-size:87px;line-height:1.02;font-weight:700;letter-spacing:-.06em;margin-bottom:13px}
.good{color:#006e62}.bad{color:#a63d28}
.meta{font-size:23px;color:#405a5d;margin-top:22px}
.mono{font-family:'DejaVu Sans Mono',monospace}
pre{margin:15px 0 0;font-size:31px;line-height:1.42;white-space:pre-wrap}
.steps{display:flex;gap:14px;margin-top:24px}
.step{flex:1;padding:16px 18px;background:#e0ebe6;border-radius:12px;font-size:22px;font-weight:600}
.focus{outline:3px solid #007e6f;outline-offset:-3px}
.muted{opacity:.26}
.cells{display:grid;grid-template-columns:repeat(9,1fr);gap:10px;margin-top:22px}
.cell{border-radius:9px;background:#e7eeeb;padding:12px;text-align:center;font-size:20px;color:#405a5d}
.cell b{display:block;font-size:35px;color:#152a2d}.cell.focus{background:#cce9df}
.compare{display:grid;grid-template-columns:186px 1fr 1fr;gap:17px 20px;align-items:center;font-size:23px}
.compare.wide{grid-template-columns:230px 1fr 1fr;gap:22px 26px}
.compare.wide .bar{height:56px}.compare.wide .bar b{top:14px}
.compare strong{font-size:25px}
.bar{position:relative;background:#e7eeeb;border-radius:8px;height:43px;overflow:hidden}
.bar i{display:block;height:100%;background:#82c5b3}.bar.red i{background:#e7a794}
.bar b{position:absolute;top:8px;left:13px}
.queue{display:grid;grid-template-columns:repeat(8,1fr);gap:5px;margin:20px 0 16px}
.queue i{display:grid;place-items:center;height:45px;border-radius:6px;background:#dce9e5;font-style:normal;font-size:20px;font-weight:660}
.queue .active{background:#006e62;color:white}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:15px 12px;border-bottom:1px solid #dbe4df;text-align:right;font-size:27px}
th:first-child,td:first-child{text-align:left}
thead th{color:#405a5d;font-size:21px}
.sub{font-size:21px;margin-top:14px}
"""


def card(tag, body, extra=""):
    return f'<div class="card {extra}"><div class="tag">{tag}</div>{body}</div>'


def stat(value, label, tag, extra=""):
    return card(tag, f'<div class="big">{value}</div><p>{label}</p>', extra)


def grid(*items):
    return f'<div class="grid {"three" if len(items) == 3 else "two"}">{"".join(items)}</div>'


def strip(*items):
    return '<div class="steps">' + "".join(f'<div class="step">{x}</div>' for x in items) + '</div>'


def show(step, first):
    return "" if step >= first else "muted"


def bar(rate, red=False):
    return f'<div class="bar {"red" if red else ""}"><i style="width:{rate:.2%}"></i><b>{rate:.0%}</b></div>'


def graph(values, corrected=None):
    xs = [90 + (n - 5) / 55 * 580 for n in SIZES]
    ys = [292 - y * 252 for y in values]
    parts = ['<svg viewBox="0 0 720 352" width="100%" style="height:352px" role="img" aria-label="Correctness by group size">']
    for rate in (0, .25, .5, .75, 1):
        y = 292 - rate * 252
        parts.append(f'<path d="M75 {y}H695" stroke="#cbdad6"/><text x="60" y="{y+6}" text-anchor="end" font-size="18" fill="#426065">{rate:.0%}</text>')
    pts = " ".join(f"{x},{y}" for x, y in zip(xs, ys))
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#a63d28" stroke-width="6"/>')
    for x, y, n in zip(xs, ys, SIZES):
        parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#a63d28"/><text x="{x}" y="325" text-anchor="middle" font-size="20" fill="#426065">{n}</text>')
    if corrected is not None:
        correct_y = [292 - y * 252 for y in corrected]
        points = " ".join(f"{x},{y}" for x, y in zip(xs, correct_y))
        parts.append(f'<polyline points="{points}" fill="none" stroke="#006e62" stroke-width="6"/>')
        for x, y in zip(xs, correct_y):
            parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="#006e62"/>')
    return "".join(parts) + "</svg>"


def result_table():
    rows = []
    for model, label in zip(MODELS, LABELS):
        values = []
        for n in SIZES:
            cell = [r for r in CORRECTED if r["model"] == model and r["num_agents"] == n]
            responses = sum(r["num_agents"] for r in cell)
            correct = sum(round(r["num_agents"] * r["correctness_rate"]) for r in cell)
            value = "100%" if correct == responses else f"{100 * correct / responses:.2f}".rstrip("0") + "%"
            values.append(f'<td class="{"good" if correct == responses else "bad"}">{value}</td>')
        rows.append(f'<tr><th scope="row">{label}</th>{"".join(values)}</tr>')
    headers = "".join(f'<th scope="col">{n}</th>' for n in SIZES)
    return f'<table><thead><tr><th scope="col">Model / requests</th>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def visual(key, step):
    if key == "drop":
        rates = [r["correctness_rate"] for r in sorted(E2, key=lambda row: row["num_agents"])]
        return grid(card("First pass · Qwen 4B · E2", graph(rates)),
                    stat(f"{rates[0]:.0%} → {rates[-1]:.0%}", "Accuracy as the group grows from 5 to 60 requests.",
                         "The apparent drop", "dark")) + strip("The first pass changed several things at once.")
    if key == "setup":
        return grid(stat("3", "Qwen 4B · Qwen 9B · DeepSeek 14B", "Pick a local model"),
                    stat("1", "Prime lookup. Each answer can be checked against a known value.", "Pick a task"),
                    stat("5–60", "Requests submitted per run on one NVIDIA DGX Spark.", "Pick a group size")) + strip(
                        "Configure", "Send one prompt per agent", "Collect answers", "Score") + (
                            '<div class="meta mono">python3 -m swarm.controls --model qwen3.5:4b</div>')
    if key == "prompt":
        primes = (2, 3, 5, 7, 11, 13, 17, 19, 23)
        cells = "".join(f'<div class="cell {"focus" if i == 8 else ""}">Index {i}<b>{p}</b></div>' for i, p in enumerate(primes))
        return card("Example prompt · shortened from Control C",
                    '<pre>You are agent 8 in a swarm of 30 agents.\nFind the prime at index 8, counting from zero.\nReply: AGENT 8 | ITEM: &lt;prime&gt;</pre>', "dark") + (
                        f'<div class="cells {show(step, 1)}">{cells}</div>')
    if key == "score":
        return card("Observed response pattern", '<pre>AGENT 8 | ITEM: 19</pre>', "dark") + '<br>' + grid(
            card("Format check", '<h2 class="good">Pass</h2><p>Valid line. Correct agent ID.</p>'),
            card("Answer check", '<h2 class="bad">Fail</h2><p>Returned 19. Expected 23.</p>', show(step, 1)))
    if key == "queue":
        serial = '<div class="queue">' + "".join(f'<i class="{"active" if i == step % 8 else ""}">{i+1}</i>' for i in range(8)) + "</div>"
        parallel = '<div class="queue">' + "".join(f'<i class="active">{i+1}</i>' for i in range(8)) + "</div>"
        return grid(card("Serial server · one slot", '<div class="big">27.37 s</div>' + serial + '<p>One request runs at a time.</p>'),
                    card("Benchmark server · 64 slots enabled", '<div class="big">7.94 s</div>' + parallel + '<p>Requests overlap.</p>', show(step, 1))) + strip(
                    "Measured wall-clock for eight requests. Slots are illustrated.")
    if key == "budget":
        # Lead with correctness: it is the single-variable control the paper
        # leads on (REPORT.md 4.1). Truncation follows, as the mechanism.
        correct, lo, hi = [], [], []
        for model, label in zip(MODELS, LABELS):
            low, high = sorted((r for r in CONTROLS if r["control"].startswith("A") and r["model"] == model), key=lambda r: r["max_tokens"])
            correct.append(f'<strong>{label}</strong>{bar(low["correctness_rate"], True)}{bar(high["correctness_rate"])}')
            lo.append(f'{low["truncated_rate"]:.0%}')
            hi.append(f'{high["truncated_rate"]:.0%}')
        return grid(stat("10% → 93%", "Qwen 4B on an identical task. Same prompt, same 30-agent swarm.",
                         "Only the token budget changed", "dark"),
                    card("Control A · measured correctness",
                         '<div class="compare wide"><span></span><strong>512 tokens</strong>'
                         '<strong>2,048 tokens</strong>' + "".join(correct) + '</div>')) + strip(
                    f'Cut off at 512 tokens · {" · ".join(lo)}', f'At 2,048 · {" · ".join(hi)}',
                    'Log <span class="mono">done_reason: length</span>')

    if key == "control":
        return grid(card("Original · index follows agent ID", '<pre>Agent  8 → index  8\nAgent 18 → index 18\nAgent 59 → index 59</pre>'),
                    card("Control C · repeat the range 0–9", '<pre>Agent  8 → index 8\nAgent 18 → index 8\nAgent 59 → index 9</pre>', show(step, 1))) + strip(
                    "Small groups still see a different mix of indices.")
    if key == "results":
        if step == 0:
            original = [r["correctness_rate"] for r in sorted(E2, key=lambda r: r["num_agents"])]
            corrected = [r["correctness_rate"] for r in sorted(
                (r for r in CORRECTED if r["model"] == MODELS[0]), key=lambda r: r["num_agents"])]
            return grid(card("Qwen 4B · accuracy by request count", graph(original, corrected)),
                        card("Two measurements of the same model",
                             '<h2 class="good">Corrected protocol · 100%</h2>'
                             '<p>Bounded prime indices, explicit indexing, and a larger budget.</p>'
                             '<br><h2 class="bad">Original protocol · 40% → 8%</h2>'
                             '<p>Higher indices entered as request count rose.</p>'))
        return card("Control C · exact-answer accuracy", result_table()) + strip(
            "DeepSeek at 50: 197 / 200", "DeepSeek at 60: 237 / 240", "Four runs at each of those sizes")
    if key == "misses":
        rows = [r for r in CORRECTED if r["model"] == MODELS[2]]
        total = sum(r["num_agents"] for r in rows)
        index_8 = sum(len(range(8, r["num_agents"], 10)) for r in rows)
        errors = [w for r in rows for w in r["wrong_answers"]]
        assert all(w["agent"] % 10 == 8 for w in errors)
        return grid(stat(f'{len(errors)} / {index_8}', 'errors at index 8', 'This item', 'dark'),
                    stat(f'0 / {total-index_8}', 'errors at all other indices', 'Remaining items')) + strip(
                    "Requested index 8", "Returned 19 · index 7", "Expected 23 · index 8")
    if key == "lure":
        total = sum(r["total_ok"] for r in E4)
        copied = sum(r["copied_wrong_answer"] for r in E4)
        return grid(card("Probe design", '<h2>Plant a wrong answer</h2><p>Give the agent a leader message containing an incorrect answer.</p><pre>AGENT 0 | ANSWER: 9999</pre>'),
                    stat(f"{copied} / {total}", "responses repeated the planted answer.", "Observed across all three models", "dark"))
    return grid(card("01 · Server", "<h2>Check overlap</h2><p>Run a batch against a serial baseline.</p>"),
                card("02 · Scorer", "<h2>Check the answer</h2><p>Try a well-formatted wrong response.</p>"),
                card("03 · Budget", "<h2>Track cutoffs</h2><p>Record the model's stop reason.</p>"),
                card("04 · Tasks", "<h2>Match the items</h2><p>Keep the task mix comparable as group size changes.</p>"))


SOURCES = {
    "drop": "Dashboard · E2 first pass", "setup": "Paper · Methods", "prompt": "Paper · Control C",
    "score": "Paper · Scoring and Control C", "queue": "Dashboard · Serialization check",
    "budget": "Dashboard · Control A", "control": "Paper · Control C design and limitations",
    "results": "Dashboard · Control C and replication", "misses": "Dashboard · Control C error items",
    "lure": "Dashboard · E4 probe", "close": "Swarm Fidelity · Research paper and results dashboard",
}


def page(scene, number, step, progress):
    label, title, _, key = scene
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<style>{CSS}</style><header><span>Swarm Fidelity · Research demo</span><span>{escape(label)}</span></header>
<main><h1>{escape(title)}</h1>{visual(key, step)}</main>
<footer><span>{SOURCES[key]}</span><span>{number:02d} / 11</span></footer>
<div class="progress"><i style="width:{progress:.1%}"></i></div></html>"""
