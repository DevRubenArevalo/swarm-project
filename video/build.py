#!/usr/bin/env python3
"""Build the 3-minute Swarm Fidelity presentation video.

  python3 video/build.py            # full build -> video/swarm_fidelity_3min.mp4
  python3 video/build.py --frames   # slides only (fast, for visual review)

Pipeline: synthesize narration -> measure real durations -> normalize to exactly
180.0s -> render slides at 1920x1080 with the measured progress rail -> stitch.
"""
import argparse, base64, csv, os, pathlib, shutil, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from narration import SLIDES                                    # noqa: E402
from slides import (CSS, INK, INK2, INK3, LINE, S1, S2, S3, BAD, GOOD, # noqa: E402
                    line_chart, bar_chart, legend)

# Scratch space for intermediate frames, WAVs and per-slide clips, and the
# directory holding the dashboard's exported datasets. Override with
# SWARM_VIDEO_WORK / SWARM_VIDEO_DATA.
WORK = pathlib.Path(os.environ.get("SWARM_VIDEO_WORK", HERE / "build"))
DASH = pathlib.Path(os.environ.get("SWARM_VIDEO_DATA", HERE / "data"))
OUT = HERE / "swarm_fidelity_3min.mp4"
TARGET = 180.0
MODELS = ["qwen3.5 4B", "qwen3.5 9B", "deepseek-r1 14B"]
COLS = {MODELS[0]: S1, MODELS[1]: S2, MODELS[2]: S3}


def load(name):
    with open(DASH / name) as f:
        return list(csv.DictReader(f))


def b64img(p):
    return "data:image/png;base64," + base64.b64encode(
        pathlib.Path(p).read_bytes()).decode()


# ------------------------------------------------------------------ slide bodies
def body_01():
    return f"""
<div class="kicker">AI Alignment Foundation &middot; Swarm Alignment Hackathon 2026</div>
<h1>Swarm fidelity:<br>where do local models break?</h1>
<div class="lede">We looked for the swarm size at which small local models stop
behaving like a frontier agent. <b style="color:{INK};font-weight:600">We did not
find one</b> &mdash; but four bugs in our own measurement harness each faked
exactly that result.</div>
<div class="body" style="justify-content:flex-end">
  <div style="height:1px;background:{LINE};margin-bottom:30px;max-width:1100px"></div>
  <div style="font-size:23px;color:{INK3};line-height:1.6">
    Testbed &nbsp;NVIDIA DGX Spark (Blackwell GB10, 128&nbsp;GB unified memory)<br>
    Models &nbsp;qwen3.5 4B &middot; qwen3.5 9B &middot; deepseek-r1 14B
    &nbsp;&middot;&nbsp; Swarm sizes &nbsp;5 to 60 concurrent agents
  </div>
</div>"""


def body_02():
    return f"""
<div class="kicker">The setup</div>
<h1>Can cheap local models work as a swarm?</h1>
<div class="lede">Every agent gets a small, independently checkable task and must
report it in a fixed format. We scale the swarm and measure how often each agent
is <i style="font-style:normal;color:{INK};font-weight:600">actually correct</i>
&mdash; not merely well-formatted.</div>
<div class="body">
  <div class="tiles">
    <div class="tile"><span class="v">3</span><span class="l">local models,
      4B&nbsp;/&nbsp;9B&nbsp;/&nbsp;14B</span></div>
    <div class="tile"><span class="v">5&ndash;60</span><span class="l">concurrent
      agents per run</span></div>
    <div class="tile"><span class="v">1</span><span class="l">DGX Spark,
      128&nbsp;GB unified memory</span></div>
    <div class="tile"><span class="v">4</span><span class="l">negative controls,
      one per artifact</span></div>
  </div>
</div>"""


def body_03():
    rows = load("e2_confounded.csv")
    xs = [5, 15, 30, 50, 60]
    series = [(m.replace("qwen3.5 ", "").replace("deepseek-r1 ", ""), COLS[m],
               [float(r["correctness"]) for r in rows if r["model"] == m])
              for m in MODELS]
    return f"""
<div class="kicker">The first result</div>
<h1>A clean collapse: <em>40% &rarr; 8%</em></h1>
<div class="body"><figure>
  {legend([(n, c) for n, c, _ in series])}
  {line_chart(series, xs, xlab="concurrent agents in the swarm",
              ylab="task correctness")}
  <div class="cap">All three decline monotonically &mdash; the 4B from 40% to 8%,
  the 14B from 80% to 7%. Exactly the break point we went looking for.</div>
</figure></div>"""


def body_04():
    return f"""
<div class="kicker">The reversal</div>
<h1>It was wrong.</h1>
<div class="body" style="justify-content:flex-start;padding-top:30px">
  <div style="font-size:42px;line-height:1.38;max-width:1480px;color:{INK2}">
    Every point on that curve was an artifact of
    <b style="color:{INK};font-weight:600">our own measurement harness</b>.
    Not one of them came from the models.
  </div>
  <div style="font-size:27px;color:{INK3};margin-top:46px;max-width:1300px;
       line-height:1.5">
    Characterising those artifacts &mdash; and building a negative control that
    catches each one &mdash; is this project's contribution.
  </div>
</div>"""


def body_05():
    a = [("1", "Silent serialization",
          "The server ran one request at a time, so a swarm of 60 was a queue of 60.",
          "Control: measured concurrency against the serialized server &rarr; 1.00"),
         ("2", "Format-only scoring",
          "We checked the shape of each answer, not whether it was right. A wrong "
          "answer scored 1.0.",
          "Control: self-test asserts deliberately wrong answers now fail"),
         ("3", "Token truncation",
          "A 512-token budget was spent inside a hidden reasoning field; agents "
          "returned an empty string.",
          "Control A: same task at 512 vs 2048 tokens"),
         ("4", "Difficulty confounded with size",
          "Agent <i>k</i> got a subtask of difficulty &prop; <i>k</i>, so a bigger "
          "swarm was also a harder test.",
          "Control C: difficulty held constant while N varies")]
    cards = "".join(f"""<div class="card"><div class="n">{n}</div><div>
      <div class="t">{t}</div><div class="d">{d}</div><div class="c">{c}</div>
    </div></div>""" for n, t, d, c in a)
    return f"""
<div class="kicker">Four artifacts, four negative controls</div>
<h1>Each one alone fakes a collapse</h1>
<div class="body"><div class="cards">{cards}</div></div>"""


def body_06():
    rows = load("serialization.csv")
    xs = [1, 2, 4, 8]
    def pick(key):
        return [float(r["concurrency"]) for r in rows if r["server"].startswith(key)]
    series = [("benchmark server (-np 64)", GOOD, pick("benchmark")),
              ("shared server (-np 1)", BAD, pick("shared"))]
    return f"""
<div class="kicker">Artifact 1 &mdash; silent serialization</div>
<h1>A &ldquo;swarm of 60&rdquo; was a <em>queue</em> of 60</h1>
<div class="body"><figure>
  {legend([(n, c) for n, c, _ in series])}
  {line_chart(series, xs, xlab="agents launched at once",
              ylab="effective concurrency \u2014 dashed line = ideal (one agent, one slot)",
              ymax=8.0, ytick=[0, 2, 4, 6, 8], yfmt=lambda v: f"{v:.0f}",
              endfmt=lambda v: f"{v:.2f}", endnames=["benchmark", "shared"],
              mr=230, ref=[(1, 1), (8, 8)])}
  <div class="cap">8 agents on the shared server took 8&times; as long as 1
  &mdash; a queue. <b style="color:{INK};font-weight:600">Both standard
  parallelism metrics score that queue as 1.00: perfect parallelism.</b>
  They are algebraically incapable of detecting the failure they are meant to catch.</div>
</figure></div>"""


def body_07():
    return f"""
<div class="kicker">Artifact 3 &mdash; token truncation</div>
<h1>83&ndash;97% of agents returned <em>nothing</em></h1>
<div class="body"><figure>
  <img src="{b64img(ROOT/'output/fig2_truncation.png')}">
  <div class="cap">At 512 tokens the budget was exhausted inside a hidden
  <span class="mono">thinking</span> field, so <span class="mono">response</span>
  came back empty &mdash; and we scored silence as a model failure. At 2048 tokens
  truncation falls to 7&ndash;17%.</div>
</figure></div>"""


def body_08():
    return f"""
<div class="kicker">The corrected result</div>
<h1>No swarm-size break point</h1>
<div class="body"><figure>
  <img src="{b64img(ROOT/'output/fig1_fidelity_curve.png')}">
  <div class="cap">Difficulty held constant (Control C). 4B and 9B hold
  <b style="color:{INK};font-weight:600">1.00</b> from N=5 to N=60; the 14B sits at
  <b style="color:{INK};font-weight:600">0.985&ndash;1.00</b>. Separately, in the
  sycophancy probe, <b style="color:{INK};font-weight:600">0 of 480</b> agents
  copied a wrong answer planted by a fake leader.</div>
</figure></div>"""


def body_09():
    idx = list(range(10))
    vals = [0.0] * 10
    vals[8] = 0.125
    cols = [INK3 if v == 0 else BAD for v in vals]
    return f"""
<div class="kicker">The residual errors</div>
<h1>Six errors, <em>one question</em></h1>
<div class="body"><figure>
  {bar_chart([str(i) for i in idx], vals, cols, h=470, ymax=0.2,
             valfmt=lambda v: f"{v:.1%}",
             xlab="prime index requested (14B, pooled over 740 responses)",
             highlight=8)}
  <div class="cap">All 6 of the 14B's errors are the same item &mdash; the 8th
  prime, answered <span class="mono">19</span> instead of
  <span class="mono">23</span>. Zero errors at every other index, at every swarm
  size. That is one fact the model has wrong, not a crowd degrading under load.</div>
</figure></div>"""


def body_10():
    return f"""
<div class="kicker">What does scale</div>
<h1>Model size caps throughput &mdash; <b>swarm size does not</b></h1>
<div class="body"><figure>
  <img src="{b64img(ROOT/'output/fig3_concurrency_ceiling.png')}" style="max-height:700px">
  <div class="cap">At N=60 the 4B sustained <b style="color:{INK};font-weight:600">35.2</b>
  genuinely concurrent agents, the 9B <b style="color:{INK};font-weight:600">36.5</b>,
  the 14B only <b style="color:{INK};font-weight:600">9.5</b> &mdash; all three at
  full accuracy. The ceiling is set by the model you choose, not by how many
  agents you ask for.</div>
</figure></div>"""


def body_11():
    rules = [("1", "Prove achieved concurrency against a <b>serial baseline</b>. "
                   "Do not trust total/max latency."),
             ("2", "Score <b>content, not format</b>. Assert that a deliberately "
                   "wrong answer fails."),
             ("3", "Record <span class='mono'>done_reason</span>. A cut-off answer "
                   "is <b>no measurement</b>, not a failure."),
             ("4", "Hold per-agent <b>difficulty constant</b> when you vary swarm size.")]
    rl = "".join(f'<div class="rule"><span>{n}</span><div>{t}</div></div>'
                 for n, t in rules)
    return f"""
<div class="kicker">What it proves</div>
<h1>The break point was in the <em>instrument</em></h1>
<div class="lede">Up to 60 concurrent agents on this hardware, small local models
did not degrade &mdash; and four plausible harness bugs each produced a convincing
curve saying they did. If you benchmark a swarm:</div>
<div class="body" style="justify-content:center;padding-bottom:40px">
  <div class="rules">{rl}</div>
  <div style="font-size:21px;color:{INK3};margin-top:40px">
    Full report, data and figures: <span class="mono">docs/REPORT.md</span>
    &nbsp;&middot;&nbsp; E1 and E4 stand as reported; E2 and E3 are retracted.
  </div>
</div>"""


BODIES = {s["id"]: f for s, f in zip(SLIDES, [
    body_01, body_02, body_03, body_04, body_05, body_06,
    body_07, body_08, body_09, body_10, body_11])}


# ------------------------------------------------------------------ pipeline
def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def synth(work):
    """Synthesize each line, return measured speech durations."""
    out = []
    for s in SLIDES:
        w = work / f"{s['id']}.wav"
        sh(["flite", "-voice", "slt", "-t", s["say"], "-o", str(w)])
        d = float(subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(w)], capture_output=True, text=True).stdout)
        out.append(d)
    return out


def plan(speech):
    """Scale the pads so speech + pads == TARGET exactly."""
    pads = [s["pad"] for s in SLIDES]
    slack = TARGET - sum(speech)
    k = slack / sum(pads)
    pads = [p * k for p in pads]
    durs = [round(a + b, 3) for a, b in zip(speech, pads)]
    durs[-1] = round(TARGET - sum(durs[:-1]), 3)
    return durs, k


def render(work, durs, frames_only=False):
    work.mkdir(parents=True, exist_ok=True)
    cum, total = 0.0, sum(durs)
    chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    for i, (s, d) in enumerate(zip(SLIDES, durs)):
        cum += d
        pct = 100.0 * cum / total
        html = (f"<!doctype html><html><head><meta charset=utf-8>"
                f"<style>{CSS}</style></head><body>"
                f'<div class="slide">{BODIES[s["id"]]()}</div>'
                f'<div class="foot"><span>Swarm Fidelity &middot; '
                f'AI Alignment Foundation &middot; Swarm Alignment 2026</span>'
                f'<span>{i+1} / {len(SLIDES)}</span></div>'
                f'<div class="rail"><i style="width:{pct:.2f}%"></i></div>'
                f"</body></html>")
        h = work / f"{s['id']}.html"
        h.write_text(html)
        sh([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
            "--force-device-scale-factor=1", "--window-size=1920,1080",
            "--virtual-time-budget=4000",
            f"--screenshot={work/(s['id']+'.png')}", f"file://{h}"],
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  frame {s['id']:14} {d:6.2f}s  rail {pct:5.1f}%")


def stitch(work, durs):
    # audio: each line + its tail of silence, concatenated
    parts = []
    for s, d in zip(SLIDES, durs):
        p = work / f"a_{s['id']}.wav"
        sh(["ffmpeg", "-y", "-v", "error", "-i", str(work / f"{s['id']}.wav"),
            "-af", f"adelay=400|400,apad,atrim=0:{d:.3f},"
                   f"aresample=48000,aformat=sample_fmts=s16:channel_layouts=stereo",
            str(p)])
        parts.append(p)
    alist = work / "audio.txt"
    alist.write_text("".join(f"file '{p}'\n" for p in parts))
    sh(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
        "-i", str(alist), "-c", "copy", str(work / "voice.wav")])

    # video: still per slide, 0.4s crossfades
    # Each xfade overlaps its two inputs by XF, so every clip is cut XF longer
    # than its slot and transition i starts at (cumulative time) - XF. Without
    # that compensation the chain finishes 10*XF short and -shortest would clip
    # the tail of the narration.
    XF = 0.4
    vs = []
    for s, d in zip(SLIDES, durs):
        v = work / f"v_{s['id']}.mp4"
        sh(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", "30",
            "-t", f"{d + XF:.3f}", "-i", str(work / f"{s['id']}.png"),
            "-vf", "scale=1920:1080,format=yuv420p", "-c:v", "libx264",
            "-preset", "medium", "-crf", "17", str(v)])
        vs.append(v)
    cmd = ["ffmpeg", "-y", "-v", "error"]
    for v in vs:
        cmd += ["-i", str(v)]
    fc, cur = [], "0:v"
    for i in range(1, len(vs)):
        off = sum(durs[:i]) - XF          # slide i is fully up at sum(durs[:i])
        lbl = f"x{i}"
        fc.append(f"[{cur}][{i}:v]xfade=transition=fade:duration={XF}:"
                  f"offset={off:.3f}[{lbl}]")
        cur = lbl
    cmd += ["-filter_complex", ";".join(fc), "-map", f"[{cur}]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-r", "30", str(work / "silent.mp4")]
    sh(cmd)

    sh(["ffmpeg", "-y", "-v", "error", "-i", str(work / "silent.mp4"),
        "-i", str(work / "voice.wav"), "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", str(OUT)])

    # captions
    def ts(t):
        h, m = int(t // 3600), int(t % 3600 // 60)
        return f"{h:02d}:{m:02d}:{t%60:06.3f}".replace(".", ",")
    srt, t = [], 0.0
    for i, (s, d) in enumerate(zip(SLIDES, durs), 1):
        srt.append(f"{i}\n{ts(t+0.4)} --> {ts(t+d)}\n{s['say']}\n")
        t += d
    (HERE / "swarm_fidelity_3min.srt").write_text("\n".join(srt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", action="store_true")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    print("synthesizing narration…")
    speech = synth(WORK)
    durs, k = plan(speech)
    print(f"speech {sum(speech):.1f}s + pads {TARGET-sum(speech):.1f}s "
          f"(pad scale {k:.2f}) = {sum(durs):.1f}s")
    print("rendering slides…")
    render(WORK, durs, a.frames)
    if a.frames:
        print(f"\nframes in {WORK}")
        return
    print("stitching…")
    stitch(WORK, durs)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", str(OUT)],
                         capture_output=True, text=True).stdout.strip()
    print(f"\n{OUT}  ({float(dur):.2f}s)")


if __name__ == "__main__":
    main()
