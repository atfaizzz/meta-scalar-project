from __future__ import annotations

from typing import Any

from env.models import Action, ToolResult
from env.reward import grade_trajectory_components
from tasks.task_easy import EASY_TASK
from tasks.task_hard import HARD_TASK
from tasks.task_medium import MEDIUM_TASK


TASKS = {
    EASY_TASK.task_id: EASY_TASK,
    MEDIUM_TASK.task_id: MEDIUM_TASK,
    HARD_TASK.task_id: HARD_TASK,
}


def _coerce_action(action_like: Action | dict[str, Any]) -> Action:
    if isinstance(action_like, Action):
        return action_like
    return Action.model_validate(action_like)


def _coerce_tool_result(tool_like: ToolResult | dict[str, Any] | None) -> ToolResult | None:
    if tool_like is None or isinstance(tool_like, ToolResult):
        return tool_like
    return ToolResult.model_validate(tool_like)


def normalize_trajectory(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for step in trajectory:
        normalized.append(
            {
                "action": _coerce_action(step["action"]),
                "tool_result": _coerce_tool_result(step.get("tool_result")),
            }
        )
    return normalized


def grade_task(task_id: str, trajectory: list[dict[str, Any]]) -> dict[str, float]:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    normalized = normalize_trajectory(trajectory)
    return grade_trajectory_components(TASKS[task_id], normalized)


def grade_easy(trajectory: list[dict[str, Any]]) -> float:
    return grade_trajectory_components(EASY_TASK, normalize_trajectory(trajectory))["score"]


def grade_medium(trajectory: list[dict[str, Any]]) -> float:
    return grade_trajectory_components(MEDIUM_TASK, normalize_trajectory(trajectory))["score"]


def grade_hard(trajectory: list[dict[str, Any]]) -> float:
    return grade_trajectory_components(HARD_TASK, normalize_trajectory(trajectory))["score"]
