"""Guards on the task suite itself.

Two of the four artifacts were defects in this file rather than in the models:
a token budget too small for a reasoning model's hidden chain-of-thought, and
a prompt template whose placeholders were never substituted. Both are cheap to
reintroduce and expensive to notice, so they are asserted here.
"""

from __future__ import annotations

import re

import pytest

from swarm.tasks import TASKS, get_task, list_tasks

TASK_IDS = [t["id"] for t in TASKS]


def test_ids_are_unique() -> None:
    assert len(TASK_IDS) == len(set(TASK_IDS))


def test_list_tasks_returns_all() -> None:
    assert len(list_tasks()) == len(TASKS)


def test_get_task_round_trips() -> None:
    for task_id in TASK_IDS:
        assert get_task(task_id)["id"] == task_id


def test_get_task_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        get_task("no_such_task")


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
class TestEveryTask:
    def test_has_the_required_fields(self, task: dict) -> None:
        for field in ("id", "name", "description", "prompt_template",
                      "expected_pattern", "max_tokens", "difficulty"):
            assert field in task, f"{task.get('id')} is missing {field}"

    def test_expected_pattern_compiles(self, task: dict) -> None:
        re.compile(task["expected_pattern"])

    def test_token_budget_fits_a_reasoning_model(self, task: dict) -> None:
        # Artifact 3. At 512 these models burned the whole budget inside a
        # hidden `thinking` field and returned "", which scored as failure.
        # Measured usage was 1800-3250 characters before the answer.
        assert task["max_tokens"] >= 2048, (
            f"{task['id']} budget is {task['max_tokens']}; below 2048 the "
            "truncation artifact returns"
        )

    def test_prompt_template_substitutes_cleanly(self, task: dict) -> None:
        # Artifact adjacent: the role task once shipped a template whose
        # placeholders survived into the prompt, so agents were told their
        # role was literally "<role>" and scored 0 for obeying.
        rendered = task["prompt_template"].format(agent_id=3, swarm_size=30)
        assert "{" not in rendered and "}" not in rendered
        assert "<role>" not in rendered
        assert "%%" not in rendered, "literal %% never collapsed to %"
        assert "3" in rendered and "30" in rendered
