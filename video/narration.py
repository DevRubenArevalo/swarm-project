"""Narration script for the 3-minute Swarm Fidelity video.

One entry per slide. `say` is the spoken text (TTS or human re-record);
`pad` is extra silence appended so the visual holds after the voice stops.
Numbers are spelled out where TTS would otherwise mangle them.
"""

SLIDES = [
    dict(id="01_title", pad=1.2, say=(
        "Swarm fidelity. Where do local models break? "
        "A measurement study on a single NVIDIA DGX Spark."
    )),
    dict(id="02_question", pad=0.8, say=(
        "The question: can cheap local models run as a swarm of agents, "
        "or do they fall apart as the swarm grows? "
        "We ran swarms of five to sixty agents against three local models, "
        "four B, nine B, and fourteen B, "
        "measuring how often each agent got its task right."
    )),
    dict(id="03_false", pad=1.0, say=(
        "Our first answer looked great. Correctness collapsed from forty percent "
        "down to eight percent as the swarm grew. Exactly the break point we went looking for."
    )),
    dict(id="04_wrong", pad=1.4, say=(
        "It was wrong. Every point on that curve came from a bug in our own "
        "measuring equipment, not from the models. "
        "Finding those bugs is what this project actually contributes."
    )),
    dict(id="05_artifacts", pad=0.8, say=(
        "There were four. One: the server ran our agents one at a time, "
        "so a swarm of sixty was really a queue of sixty. "
        "Two: our scorer checked the shape of each answer, not whether it was right. "
        "Three: our token budget cut agents off mid-thought, and we scored that as failure. "
        "Four: agent fifty got a harder problem than agent five, "
        "so a bigger swarm was also a harder test. "
        "Each one alone produces a fake collapse."
    )),
    dict(id="06_serial", pad=0.8, say=(
        "Here is the proof. On the old server, eight agents took "
        "eight times as long as one. Effective concurrency, one point zero. A queue. "
        "On the fixed server, eight agents finished in the time of two. "
        "Concurrency, seven point two. "
        "And here is what matters: both standard parallelism formulas score that "
        "queue as perfect parallelism. They cannot see the failure."
    )),
    dict(id="07_trunc", pad=0.8, say=(
        "The token budget was just as bad. At five hundred and twelve tokens, "
        "eighty three to ninety seven percent of agents ran out of room inside a "
        "hidden reasoning field and returned nothing. "
        "We were scoring silence as stupidity. Raise the budget, and it nearly vanishes."
    )),
    dict(id="08_corrected", pad=1.0, say=(
        "Fix all four, hold each agent's difficulty constant, and this is the result. "
        "Flat. From five agents to sixty, the four B and nine B are perfect, "
        "and the fourteen B sits at ninety nine percent. No break point. "
        "And not one agent in four hundred and eighty copied a wrong answer "
        "planted by a fake leader."
    )),
    dict(id="09_oneitem", pad=0.8, say=(
        "The fourteen B's errors are not a scale effect. All six are the same question, "
        "the eighth prime, answered nineteen instead of twenty three. "
        "Every other question, at every swarm size, was right. "
        "That is one fact the model has wrong, not a crowd getting confused."
    )),
    dict(id="10_ceiling", pad=0.8, say=(
        "What does scale with model size is throughput. The four B held thirty five "
        "agents truly in parallel, the nine B thirty six, the fourteen B only nine. "
        "Model size sets your ceiling. Swarm size does not."
    )),
    dict(id="11_close", pad=1.6, say=(
        "The finding: on this hardware, small local models did not degrade as the "
        "swarm grew. Four plausible measurement bugs each faked that result. "
        "So prove your concurrency against a serial baseline. Score content, not format. "
        "Treat a cut-off answer as no data. Hold difficulty constant. "
        "Otherwise you will find a break point that is not there."
    )),
]

if __name__ == "__main__":
    total = sum(len(s["say"].split()) for s in SLIDES)
    print(f"{len(SLIDES)} slides, {total} spoken words")
    for s in SLIDES:
        print(f"  {s['id']:14} {len(s['say'].split()):4d} w")
