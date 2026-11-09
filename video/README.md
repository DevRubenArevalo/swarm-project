# 3-minute presentation video

**Final cut:** `swarm_fidelity_demo_natural.mp4` — 1920×1080, 30 fps,
**180.00 s**, 5.7 MB. Captions: `swarm_fidelity_demo_natural.srt`.

Two renderers live here. `build.py` + `slides.py` + `narration.py` is the
original deck (stills in `slides/`); `demo/` is the revised cut that produced
the final video, with its own layout engine and a neural narration voice
(`demo/revoice.py`). Both assemble to exactly 180.000 s by the same method.

Eleven slides, cut from the dashboard's own datasets
(`e1_scaling`, `e2_confounded`, `control_ab`, `control_c`, `serialization`,
`error_items`) and the three report figures, so every number on screen traces
to a row in `output/`.

| # | Slide | Carries |
|---|---|---|
| 1 | Title | testbed, models, swarm sizes |
| 2 | The setup | 3 models · 5–60 agents · 1 Spark · 4 controls |
| 3 | The first result | the false collapse, 4B 40%→8%, 14B 80%→7% |
| 4 | The reversal | the claim retracted |
| 5 | Four artifacts | each bug + its negative control |
| 6 | Artifact 1 | concurrency 1.01 (queue) vs 7.18 (parallel) |
| 7 | Artifact 3 | truncation 90/97/83% at 512 → 7/13/17% at 2048 |
| 8 | The corrected result | flat curve; 0 of 480 sycophancy |
| 9 | Residual errors | 6 errors, all prime index 8, 12.5% on that item |
| 10 | Concurrency ceiling | 35.2 / 36.5 / 9.5 at N=60 |
| 11 | What it proves | the four rules |

## Rebuild

```bash
python3 video/build.py            # full build
python3 video/build.py --frames   # slides only, for visual review
```

Timing is **measured, not guessed**: `build.py` synthesizes each line, reads the
real duration back with `ffprobe`, then scales the inter-slide pauses so the
total lands on exactly 180.000 s. Edit `video/narration.py` and rebuild — the
slide durations, the progress rail and the `.srt` all follow automatically.

## Replacing the voice

Narration is synthesized offline — `flite -voice slt` in `build.py`, a Kokoro
ONNX voice in `demo/revoice.py`. Neither needs network access. To drop in a
human recording, read `narration.py` aloud against the timings the build
prints, then:

```bash
ffmpeg -i video/swarm_fidelity_demo_natural.mp4 -i yourvoice.wav \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 160k -shortest \
  video/swarm_fidelity_vo.mp4
```

The deck is designed to read **silent** — every number spoken is also on screen —
so muting the track costs nothing if you would rather present it live.
