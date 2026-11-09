"""Render the Swarm Fidelity demo with local Piper narration and timed captions.

Requires ffmpeg, ffprobe, Chrome and piper-tts. Pass the Piper ONNX voice path
with --voice. The voice's matching .onnx.json file must sit beside it.
"""
import argparse
from itertools import accumulate
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import wave

from layout import page
from scenes import SCENES

OUT = Path(__file__).resolve().parents[1]
STEM = "swarm_fidelity_demo_3min"
FPS = 30
TARGET = 180
FRAMES = TARGET * FPS


def run(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def synthesize(work, model):
    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(model)
    cues = []
    for i, scene in enumerate(SCENES, 1):
        sentences = re.split(r"(?<=[.!?])\s+", scene[2])
        for j, sentence in enumerate(sentences):
            path = work / f"voice-{i:02d}-{j}.wav"
            with wave.open(str(path), "wb") as f:
                voice.synthesize_wav(sentence, f, SynthesisConfig(length_scale=1.02))
            with wave.open(str(path), "rb") as f:
                duration = f.getnframes() / f.getframerate()
            cues.append((i, j, sentence, path, duration, j == len(sentences) - 1))
    return cues


def timing(cues):
    speech = sum(c[4] for c in cues)
    pauses = [.95 if c[5] else .19 for c in cues]
    tempo = speech / (TARGET - sum(pauses) - 1.7)
    if not .78 <= tempo <= 1.25:
        raise ValueError(f"Narration needs a script edit to fit 180 s, required tempo {tempo:.2f}")
    end = [round(t * FPS) for t in accumulate(
        c[4] / tempo + pause for c, pause in zip(cues, pauses))]
    end[-1] = FRAMES
    return [0, *end], tempo


def stamp(seconds):
    ms = round(seconds * 1000)
    whole, fraction = divmod(ms, 1000)
    return f"{whole//3600:02}:{whole//60%60:02}:{whole%60:02},{fraction:03}"


def screenshot(chrome, work, scene, idx, state, progress):
    base = work / f"scene-{idx:02d}-{state}"
    html, png = base.with_suffix(".html"), base.with_suffix(".png")
    html.write_text(page(scene, idx, state, progress))
    run(chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
        "--force-device-scale-factor=1", "--window-size=1600,900",
        f"--user-data-dir={work / 'chrome'}", f"--screenshot={png}",
        html.as_uri(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return png


def state_for(key, sentence):
    """Time the diagram changes to the matching words."""
    reveal_at = {"prompt": 2, "score": 2, "queue": 2, "control": 1, "results": 2}
    return int(key in reveal_at and sentence >= reveal_at[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome is None:
        parser.error("Chrome or Chromium is required")

    with tempfile.TemporaryDirectory(prefix="swarm-demo-") as temp:
        work = Path(temp)
        cues = synthesize(work, args.voice)
        bounds, tempo = timing(cues)
        print(f"{len(cues)} narration cues. Tempo {tempo:.2f}.", flush=True)
        shots, media, captions = {}, [], []
        for index, cue in enumerate(cues):
            number, sentence_no, sentence, wav, spoken, _ = cue
            scene = SCENES[number - 1]
            state = state_for(scene[3], sentence_no)
            key = (number, state)
            if key not in shots:
                shots[key] = screenshot(chrome, work, scene, number, state, bounds[index] / FRAMES)
                if args.preview_dir:
                    args.preview_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(shots[key], args.preview_dir)
                print(f"{bounds[index]/FPS:6.1f}s  {scene[0]} / {state}", flush=True)

            frames = bounds[index + 1] - bounds[index]
            duration = frames / FPS
            clip = work / f"clip-{index:02d}.mkv"
            run("ffmpeg", "-nostdin", "-v", "error", "-y", "-loop", "1",
                "-framerate", FPS, "-i", shots[key], "-i", wav,
                "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p",
                "-af", f"atempo={tempo:.8f},aresample=48000,adelay=60,apad,atrim=duration={duration:.8f}",
                "-t", f"{duration:.8f}", "-c:v", "libx264", "-threads", "3", "-preset",
                "veryfast", "-crf", "17", "-c:a", "pcm_s16le", clip)
            media.append(f"file '{clip.name}'\n")
            start = bounds[index] / FPS + .06
            end = min(bounds[index + 1] / FPS - .02, start + spoken / tempo + .08)
            captions.append(f"{index+1}\n{stamp(start)} --> {stamp(end)}\n{textwrap.fill(sentence, width=46)}\n")

        playlist = work / "clips.txt"
        playlist.write_text("".join(media))
        subtitles = OUT / f"{STEM}.srt"
        subtitles.write_text("\n".join(captions))
        run("ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0",
            "-i", playlist, "-i", subtitles, "-map", "0:v", "-map", "0:a", "-map", "1:s",
            "-vf", f"setpts=N/({FPS}*TB)", "-r", FPS,
            "-c:v", "libx264", "-threads", "4", "-preset", "veryfast", "-crf", "18",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000,pan=stereo|c0=c0|c1=c0,apad",
            "-c:a", "aac", "-b:a", "160k", "-c:s", "mov_text", "-metadata:s:s:0",
            "language=eng", "-disposition:s:0", "0", "-t", TARGET, "-movflags", "+faststart",
            OUT / f"{STEM}.mp4")


if __name__ == "__main__":
    main()
