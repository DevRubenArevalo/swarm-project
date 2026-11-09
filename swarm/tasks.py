"""Task suite: 4 tasks of increasing complexity for swarm benchmarking."""

from __future__ import annotations


TASKS: list[dict] = [
    {
        "id": "identity",
        "name": "Identity Check",
        "description": "Agent must reply with its ID, status, and swarm size. Tests basic instruction following.",
        "prompt_template": (
            "You are agent {agent_id} in a swarm of {swarm_size} agents. "
            "Reply with exactly one line in this format:\n"
            "AGENT {agent_id} | STATUS: ready | SWARM: {swarm_size}\n"
            "Do not add anything else."
        ),
        "expected_pattern": r"AGENT\s+\d+\s*\|\s*STATUS:\s*ready\s*\|\s*SWARM:\s+\d+",
        "max_tokens": 2048,  # >=2048: thinking models emit 1800-3250 chars before answering (512 truncated 90% of agents)
        "difficulty": 1,
    },
    {
        "id": "math_relay",
        "name": "Math Relay",
        "description": "Each agent computes (agent_id * swarm_size) + 42. Tests arithmetic under swarm context.",
        "prompt_template": (
            "You are agent {agent_id} in a swarm of {swarm_size} agents. "
            "Compute the value: (agent_id × swarm_size) + 42, where agent_id = {agent_id} and swarm_size = {swarm_size}. "
            "Reply with exactly one line:\n"
            "AGENT {agent_id} | ANSWER: <number>\n"
            "Do not add anything else."
        ),
        "expected_pattern": r"AGENT\s+\d+\s*\|\s*ANSWER:\s*\d+",
        "max_tokens": 2048,  # >=2048: thinking models emit 1800-3250 chars before answering (512 truncated 90% of agents)
        "difficulty": 2,
    },
    {
        "id": "role_adherence",
        "name": "Role Adherence",
        "description": "Each agent is assigned a role (planner/builder/reviewer) based on agent_id % 3. Tests role persistence.",
        "prompt_template": (
            # Fixed 2026-11-07: the template used `.replace()`, not %-formatting,
            # so the literal "%%" never collapsed to "%" and the agent read
            # "agent_id %% 3". Worse, "<role>" was never substituted, so every
            # agent was told its role was literally "<role>" and then asked to
            # echo it back — while the regex demanded planner|builder|reviewer.
            # Agents that obeyed scored 0. The task now states the rule once and
            # asks the agent to derive its own role, which is the actual test.
            "You are agent {agent_id} in a swarm of {swarm_size} agents. "
            "Determine your role by computing agent_id mod 3, where "
            "0=planner, 1=builder, 2=reviewer. Your agent_id is {agent_id}. "
            "Reply with exactly one line, substituting your real role and a real action:\n"
            "AGENT {agent_id} | ROLE: <your role> | ACTION: <one sentence describing what that role would do next>\n"
            "Do not add anything else. Do not output the angle brackets."
        ),
        "expected_pattern": r"AGENT\s+\d+\s*\|\s*ROLE:\s*(planner|builder|reviewer)\s*\|\s*ACTION:\s*.+",
        "max_tokens": 2048,  # >=2048: thinking models emit 1800-3250 chars before answering (512 truncated 90% of agents)
        "difficulty": 3,
    },
    {
        "id": "coordinated_output",
        "name": "Coordinated Output",
        "description": "Each agent must produce the i-th smallest prime. Tests global ordering constraint without seeing other agents.",
        "prompt_template": (
            "You are agent {agent_id} in a swarm of {swarm_size} agents. "
            "Collectively, the swarm must produce the first {swarm_size} prime numbers in order. "
            "Your job: reply with the {agent_id}-th smallest prime number (0-indexed, so agent 0 replies with 2, agent 1 with 3, agent 2 with 5, etc.). "
            "Reply with exactly one line:\n"
            "AGENT {agent_id} | ITEM: <prime>\n"
            "Do not add anything else."
        ),
        "expected_pattern": r"AGENT\s+\d+\s*\|\s*ITEM:\s*\d+",
        "max_tokens": 2048,  # >=2048: thinking models emit 1800-3250 chars before answering (512 truncated 90% of agents)
        "difficulty": 4,
    },
]


def get_task(task_id: str) -> dict:
    """Get a task by its ID."""
    for t in TASKS:
        if t["id"] == task_id:
            return t
    raise ValueError(f"Unknown task: {task_id}. Available: {[t['id'] for t in TASKS]}")


def list_tasks() -> list[dict]:
    """List all available tasks."""
    return TASKS


if __name__ == "__main__":
    print("Available tasks:")
    for t in list_tasks():
        print(f"  [{t['difficulty']}] {t['id']:20s} — {t['name']}")
        print(f"       {t['description']}")
