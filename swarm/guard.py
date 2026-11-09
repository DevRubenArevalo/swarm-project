"""Clean-run guard: refuse to benchmark while unexpected models share the GPU.

Ollama keeps loaded models in VRAM for a while (default keep_alive=5m) and
serves concurrent requests from the same GPU. If a model we did NOT ask for is
resident, our latency/completion numbers are contaminated by contention.

This module checks /api/ps before a run and blocks (or warns) if the set of
loaded models is not a subset of the models we intend to use.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

OLLAMA = os.environ.get("SWARM_OLLAMA_URL", "http://localhost:11435")
# The shared instance, checked for co-residency even when we benchmark on another
# port: both servers share one GPU, so a model loaded there still contends.
SHARED_OLLAMA = os.environ.get("SWARM_SHARED_OLLAMA_URL", "http://localhost:11434")


def loaded_models(ollama_url: str = OLLAMA) -> list[str]:
    """Return the names of models currently resident in VRAM."""
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/ps", timeout=5) as r:
            data = json.loads(r.read())
        return [m.get("name", "?") for m in data.get("models", [])]
    except Exception as e:
        raise RuntimeError(f"Could not query Ollama at {ollama_url}: {e}")


def is_clean(loaded: list[str], allowed: list[str]) -> bool:
    """True if every loaded model is in the allowed set (or none are loaded)."""
    allowed_set = {a.split(":")[0] for a in allowed}  # ignore tag for matching
    for name in loaded:
        base = name.split(":")[0]
        if base not in allowed_set:
            return False
    return True


def assert_clean(
    allowed: list[str],
    ollama_url: str = OLLAMA,
    wait: bool = False,
    max_wait_s: int = 120,
) -> list[str]:
    """
    Block until the GPU is clean (only `allowed` models resident), or raise.

    Args:
        allowed: model names we intend to use this run.
        wait: if True, poll until clean or max_wait_s elapses.
        max_wait_s: how long to wait when wait=True.

    Returns:
        The list of loaded models at the moment we cleared.

    Raises:
        RuntimeError: if a foreign model is resident and wait=False (or timeout).
    """
    deadline = time.monotonic() + max_wait_s
    while True:
        loaded = loaded_models(ollama_url)
        if is_clean(loaded, allowed):
            return loaded
        if not wait or time.monotonic() > deadline:
            foreign = [m for m in loaded if m.split(":")[0] not in {a.split(":")[0] for a in allowed}]
            raise RuntimeError(
                f"GPU not clean — foreign model(s) resident: {foreign}. "
                f"Allowed: {allowed}. Refusing to run to avoid contaminated data."
            )
        time.sleep(3)


def residency(urls: list[str] | None = None) -> list[str]:
    """
    Every model resident on the GPU, LABELLED BY INSTANCE.

    Returns entries like "bench/qwen3.5:4b" or "shared/qwen3.5:4b".

    The label matters. Both Ollama instances share one physical GPU, and an
    unlabelled name cannot distinguish "the 4B we are benchmarking on :11435"
    from "someone else's 4B chat session on :11434". Those are different
    processes with different VRAM and different compute load, but identical
    model names — so an unlabelled check silently reports a clean run while
    another user's session of the SAME model competes for the GPU.

    Recorded on each result row so contention is a filterable data column
    rather than an invisible confound. Unreachable instances are skipped.
    """
    if urls is None:
        pairs = [(OLLAMA, "bench"), (SHARED_OLLAMA, "shared")]
    else:
        pairs = [(u, u.rsplit(":", 1)[-1]) for u in dict.fromkeys(urls)]
    seen: list[str] = []
    for url, label in pairs:
        try:
            for name in loaded_models(url):
                entry = f"{label}/{name}"
                if entry not in seen:
                    seen.append(entry)
        except RuntimeError:
            continue
    return seen


def model_base(entry: str) -> str:
    """Base model name from a plain name or a labelled residency entry."""
    return entry.rsplit("/", 1)[-1].split(":")[0]


def contention(entries: list[str], model_under_test: str) -> list[str]:
    """
    The subset of residency entries that could contaminate a measurement.

    Anything on the shared instance counts — we never benchmark there, so any
    model loaded there belongs to someone else and competes for the GPU, even
    when it happens to share our model's name. On our own instance, only a
    model other than the one under test counts.
    """
    mb = model_base(model_under_test)
    out = []
    for e in entries:
        if e.startswith("shared/") or model_base(e) != mb:
            if e not in out:
                out.append(e)
    return out


def main() -> int:
    allowed = sys.argv[1:] if len(sys.argv) > 1 else ["qwen3.5:4b", "qwen3.5:9b", "deepseek-r1:14b"]
    try:
        loaded = assert_clean(allowed, wait=True, max_wait_s=120)
        print(f"✅ CLEAN — resident models: {loaded or '(none)'}")
        return 0
    except RuntimeError as e:
        print(f"⛔ BLOCKED — {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
