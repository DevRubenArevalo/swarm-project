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
| 1 | The drop | the false collapse, 40% → 8% |
| 2 | Set up a run | 3 models · 5–60 requests · one DGX Spark |
| 3 | Follow agent 8 | one prompt with an independently checkable answer |
| 4 | Check the score | format check passes, answer check fails (19 vs 23) |
| 5 | Check the server | 27.37 s serial vs 7.94 s parallel, same 8 requests |
| 6 | **A budget limit, not a model limit** | **Control A: correctness 10% → 93%**, cutoffs 90/97/83% → 7/13/17% |
| 7 | Change one variable | Control C index cycling |
| 8 | Read the dashboard | the corrected curve; the steep drop is gone |
| 9 | Inspect the misses | 6 misses, all prime index 8 |
| 10 | Run a separate probe | 0 of 480 copied the planted answer |
| 11 | Take it into your next run | the four checks |

Slide 6 carries the paper's lead evidence (`docs/REPORT.md` §4.1): the
single-variable control in which only the per-agent token budget changes.

## Rebuild

```bash
# Original deck (flite narration)
python3 video/build.py            # full build
python3 video/build.py --frames   # slides only, for visual review

# Revised cut — the one that ships. Renders the picture, then the voice.
python3 video/demo/build.py --voice /path/to/piper-voice.onnx
python3 video/demo/revoice.py /path/to/kokoro-model-dir   # voice: af_heart @ 0.98
```

`revoice.py` re-narrates against the existing picture cuts, so editing a line
in its `BEATS` table and re-running it changes the narration and captions
without touching the slides. It fits each beat into the window its scene
already occupies and fails loudly if a line cannot fit.

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
