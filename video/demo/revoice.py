"""Render a new narration against the existing picture cuts.

Install kokoro-onnx, then pass a directory containing the Kokoro v1.0 int8
model and voices. The original demo and captions supply the visual beat times.
"""
import argparse
import subprocess
import tempfile
import textwrap
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
from build import stamp
from kokoro_onnx import Kokoro

OUT = Path(__file__).resolve().parents[1]
ORIGINAL = "swarm_fidelity_demo_3min"
NEW = "swarm_fidelity_demo_natural"
RATE = 24_000

# Each beat starts at an existing picture cut. Render each phrase as one
# utterance so its punctuation and rhythm carry through into the new captions.
BEATS = [
    (1, (
        "Forty percent to eight. If your benchmark dropped that far, would you switch models?",
        "That was the story our first chart told. But before we blamed the model, we followed one answer through the test.",
    )),
    (6, (
        "Here's the setup. On one D G X Spark, we picked a local model and gave each agent a small job.",
        "Then we changed how many requests we sent, from five to sixty. We can check every answer against a known value. Let's follow one request.",
    )),
    (11, (
        "Agent eight has a simple job. Find the prime at index eight, starting from zero.",
    )),
    (13, (
        "Count along the row. At index eight, you land on twenty-three.",
        "Now suppose the agent sends back nineteen.",
    )),
    (16, (
        "At a glance, that line looks fine. Right agent number, right format.",
    )),
    (18, (
        "Here's the catch. Nineteen is the prime one slot earlier.",
        "The format check passes, but the answer check fails. Feed that exact line to your scorer and see which check it runs.",
    )),
    (21, (
        "Next we sent eight requests together. The server quietly queued them, one at a time.",
    )),
    (23, (
        "With parallel slots turned on, that batch went from about twenty-seven seconds to eight.",
        "Same number of requests. Very different overlap. That's why we check what the server actually did before we interpret the accuracy curve.",
    )),
    (26, (
        "Then there were the blank answers. Where did they go?",
        "The model used up its token budget while reasoning, and stopped before giving an answer.",
        "We raised the cap from five hundred twelve to two thousand forty-eight.",
        "On the same task, same prompt, same thirty agents, correctness went from ten percent to ninety-three.",
    )),
    (31, (
        "One wrinkle.",
    )),
    (32, (
        "The bigger groups were getting harder questions. So we cycled through indices zero to nine.",
        "Now agents eight and eighteen both get index eight. Smaller groups can still see a different mix.",
    )),
    (36, (
        "Look what happened when we reran it. The steep drop on that first chart disappeared.",
    )),
    (38, (
        "Both Qwen models got every answer right. Deep Seek stayed at ninety-eight point five percent or above.",
        "That held through sixty requests on this task.",
    )),
    (40, (
        "The remaining misses tell their own story. There were six, and every one asked for index eight.",
        "Each time Deep Seek said nineteen instead of twenty-three.",
        "A group of five never gets that item. A group of sixty sees it six times. You can see why the individual answers matter as much as the average.",
    )),
    (46, (
        "We also tried planting a wrong answer in a leader message.",
        "None of the four hundred eighty responses repeated it. That's the result of this particular test.",
    )),
    (49, (
        "If your curve falls like ours did, check four things before you switch models.",
        "Are the requests overlapping? Does the scorer catch a wrong answer?",
        "Did the model hit its token limit? Are you comparing the same mix of tasks?",
        "The paper and dashboard walk through all four.",
    )),
]


def seconds(timestamp):
    hours, minutes, tail = timestamp.replace(",", ".").split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(tail)


def original_starts():
    captions = (OUT / f"{ORIGINAL}.srt").read_text().strip().split("\n\n")
    return {
        int(block.splitlines()[0]): round(
            (seconds(block.splitlines()[1].split(" --> ")[0]) - 0.060) * 30
        ) / 30
        for block in captions
    }


def use_system_espeak():
    """Prefer a system espeak-ng.

    The espeakng_loader wheel ships a libespeak-ng.so with its build-time data
    directory compiled in, so it looks for phontab under a path that only
    existed on the machine that built the wheel. Pointing the loader at a
    system install avoids that.
    """
    import espeakng_loader

    for lib, data in (("/usr/lib64/libespeak-ng.so.1", "/usr/share/espeak-ng-data"),
                      ("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1", "/usr/share/espeak-ng-data")):
        if Path(lib).exists() and (Path(data) / "phontab").is_file():
            espeakng_loader.get_library_path = lambda value=lib: value
            espeakng_loader.get_data_path = lambda value=data: value
            return


def load_engine(root):
    use_system_espeak()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    model = ort.InferenceSession(
        str(root / "kokoro-v1.0.int8.onnx"), options, providers=["CPUExecutionProvider"],
    )
    return Kokoro.from_session(model, str(root / "voices-v1.0.bin"))


def render(engine, voice):
    starts = original_starts()
    edges = [starts[cue] for cue, _ in BEATS] + [180.0]
    track = np.zeros(180 * RATE, dtype=np.float32)
    captions = []

    for (_cue, phrases), start, end in zip(BEATS, edges, edges[1:]):
        start_at = round((start + .10) * RATE)
        end_at = round((end - (.9 if end == 180.0 else .26)) * RATE)
        budget = (end_at - start_at) / RATE
        gaps = [.17] * (len(phrases) - 1)
        speed = .98

        for _ in range(5):
            parts = [engine.create(text, voice=voice, speed=speed)[0] for text in phrases]
            duration = sum(len(part) for part in parts) / RATE + sum(gaps)
            if duration <= budget:
                break
            speed *= duration / budget * 1.025
        else:
            raise ValueError(f"Narration at {start:.2f}s overruns its scene")
        if speed > 1.23:
            print(f"Quick delivery at {start:.2f}s: {speed:.2f}×", flush=True)

        # Give the words room to settle and distribute leftover room across the
        # phrase breaks. A longer visual hold belongs at the end of the beat.
        if gaps:
            extra = min(.30, (budget - duration) / (len(gaps) + 1))
            gaps = [gap + extra for gap in gaps]

        cursor = start_at
        for text, part, pause in zip(phrases, parts, [*gaps, 0]):
            stop = cursor + len(part)
            track[cursor:stop] = part
            wrapped = textwrap.fill(text, width=49)
            captions.append(
                f"{len(captions)+1}\n{stamp(cursor/RATE)} --> {stamp(stop/RATE)}\n{wrapped}\n"
            )
            cursor = stop + round(pause * RATE)
        print(f"{start:6.2f}–{end:6.2f}  speed {speed:.2f}  hold {end-cursor/RATE:.2f}", flush=True)

    return track, "\n".join(captions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--voice", default="af_heart")
    args = parser.parse_args()
    engine = load_engine(args.model_dir)
    track, captions = render(engine, args.voice)

    with tempfile.TemporaryDirectory(prefix="swarm-revoice-") as directory:
        tmp = Path(directory)
        audio, subs = tmp / "narration.wav", tmp / "captions.srt"
        with wave.open(str(audio), "wb") as wav:
            wav.setparams((1, 2, RATE, 0, "NONE", "not compressed"))
            wav.writeframes(np.clip(track * 32767, -32768, 32767).astype("<i2").tobytes())
        subs.write_text(captions)

        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-i", str(OUT / f"{ORIGINAL}.mp4"), "-i", str(audio), "-i", str(subs),
            "-map", "0:v:0", "-map", "1:a:0", "-map", "2:s:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000,pan=stereo|c0=c0|c1=c0",
            "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
            "-disposition:s:0", "0", "-t", "180", "-movflags", "+faststart",
            str(OUT / f"{NEW}.mp4"),
        ], check=True)
        (OUT / f"{NEW}.srt").write_text(captions)


if __name__ == "__main__":
    main()
