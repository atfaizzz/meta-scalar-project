from __future__ import annotations

from typing import Any

from env.models import Action, Reward, TaskConfig, ToolResult


def _normalized(value: float, upper_bound: float) -> float:
    if upper_bound <= 0:
        return 0.0
    return max(0.0, min(1.0, value / upper_bound))


def _next_expected_action(task: TaskConfig, prior_actions: list[Action]) -> str | None:
    index = 0
    for prior in prior_actions:
        if index < len(task.expected_plan) and prior.action_type == task.expected_plan[index]:
            index += 1
    if index >= len(task.expected_plan):
        return None
    return task.expected_plan[index]


def score_tool_usage(task: TaskConfig, prior_actions: list[Action], action: Action) -> tuple[float, list[str]]:
    notes: list[str] = []
    if action.action_type == "reply_to_user":
        return 0.0, notes

    next_expected = _next_expected_action(task, prior_actions)
    if next_expected == action.action_type:
        notes.append("Used the next expected tool or workflow action.")
        return 0.2, notes

    if action.action_type in task.expected_plan:
        notes.append("Used a relevant tool, but not in the ideal order.")
        return 0.1, notes

    notes.append("Used an action that does not advance the expected workflow.")
    return 0.0, notes


def score_reasoning(task: TaskConfig, action: Action, tool_result: ToolResult | None) -> tuple[float, list[str]]:
    notes: list[str] = []
    text = " ".join(
        part
        for part in [
            action.reasoning,
            action.message or "",
            "" if tool_result is None else tool_result.summary,
        ]
        if part
    ).lower()

    marker_hits = sum(1 for marker in task.reasoning_markers if marker.lower() in text)
    score = round(0.3 * _normalized(marker_hits, max(1, min(3, len(task.reasoning_markers)))), 4)
    if score > 0:
        notes.append("Reasoning references task-relevant evidence or policy details.")
    elif action.reasoning.strip():
        notes.append("Reasoning was present, but it missed task-specific evidence.")

    return score, notes


def score_resolution(
    task: TaskConfig,
    prior_actions: list[Action],
    action: Action,
    tool_result: ToolResult | None,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    text = " ".join(part for part in [action.message or "", action.escalation_note or ""] if part).lower()

    if action.action_type == "escalate" and task.escalation_required:
        score = 0.15
        if "manager" in text or "review" in text or "exception" in text:
            score = 0.18
        notes.append("Created an escalation path for a policy exception.")
        return score, notes

    if action.action_type == "issue_refund" and tool_result is not None:
        approved = bool(tool_result.data.get("approved"))
        if task.escalation_required and not approved:
            notes.append("Confirmed that the refund tool requires escalation beyond policy.")
            return 0.12, notes
        if approved:
            notes.append("Issued a valid refund or exception credit.")
            return 0.14, notes
        return 0.0, notes

    if action.action_type != "reply_to_user":
        return 0.0, notes

    keyword_hits = sum(1 for keyword in task.resolution_keywords if keyword.lower() in text)
    raw_score = 0.3 * _normalized(keyword_hits, max(2, len(task.resolution_keywords)))

    if task.order_id and not any(previous.action_type == "check_order_status" for previous in prior_actions):
        raw_score *= 0.6
        notes.append("Reply attempted a resolution before confirming the order state.")

    if task.preferred_kb_article and not any(previous.action_type == "query_kb" for previous in prior_actions):
        raw_score *= 0.6
        notes.append("Reply did not ground the answer in the knowledge base.")

    if task.escalation_required and not any(previous.action_type == "escalate" for previous in prior_actions):
        raw_score *= 0.3
        notes.append("Reply references a policy exception without escalating the case.")

    if raw_score > 0:
        notes.append("Final resolution addresses the main customer need.")

    return round(raw_score, 4), notes


def score_clarity(action: Action) -> tuple[float, list[str]]:
    notes: list[str] = []
    text = (action.message or action.reasoning or action.escalation_note or "").strip()
    if not text:
        return 0.0, notes

    word_count = len(text.split())
    score = 0.0
    if 8 <= word_count <= 80:
        score += 0.1
    if any(token in text.lower() for token in ["sorry", "thanks", "update", "review", "help", "confirm"]):
        score += 0.1

    if score > 0:
        notes.append("Response is concise and clear.")

    return round(min(score, 0.2), 4), notes


def score_penalties(
    task: TaskConfig,
    prior_actions: list[Action],
    action: Action,
    tool_result: ToolResult | None,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    penalty = 0.0

    if action.action_type == "reply_to_user" and not action.message:
        penalty += 0.2
        notes.append("Invalid action: reply_to_user requires a non-empty message.")

    if action.action_type == "query_kb" and not (action.query or action.message):
        penalty += 0.15
        notes.append("Invalid action: query_kb requires a search query.")

    if action.action_type == "issue_refund":
        if action.refund_amount is None:
            penalty += 0.15
            notes.append("Invalid action: issue_refund should include a refund amount.")
        elif action.refund_amount <= 0:
            penalty += 0.2
            notes.append("Invalid action: refund amount must be greater than zero.")

    duplicate_tools = [step.action_type for step in prior_actions if step.action_type == action.action_type]
    if action.action_type != "reply_to_user" and duplicate_tools:
        penalty += 0.05
        notes.append("Repeated a tool action without moving the case forward.")

    reply_text = (action.message or "").lower()
    if task.order_id and action.action_type == "reply_to_user":
        mentions_tracking_fact = any(token in reply_text for token in ["eta", "tracking", "weather", "delivered"])
        has_order_lookup = any(step.action_type == "check_order_status" for step in prior_actions)
        if mentions_tracking_fact and not has_order_lookup:
            penalty += 0.15
            notes.append("Hallucination risk: reply mentions order facts before checking the order.")

    if task.escalation_required and action.action_type == "reply_to_user":
        mentions_approval = "refund approved" in reply_text or "processed the refund" in reply_text
        approved_refund = tool_result is not None and bool(tool_result.data.get("approved"))
        prior_refund_approval = any(
            step.action_type == "issue_refund" for step in prior_actions
        ) and approved_refund
        if mentions_approval and not prior_refund_approval:
            penalty += 0.2
            notes.append("Hallucination risk: promised a refund approval that the system did not confirm.")

    return round(penalty, 4), notes


def compute_step_reward(
    task: TaskConfig,
    prior_actions: list[Action],
    action: Action,
    tool_result: ToolResult | None,
) -> Reward:
    rationale: list[str] = []
    tool_usage, notes = score_tool_usage(task, prior_actions, action)
    rationale.extend(notes)

    reasoning, notes = score_reasoning(task, action, tool_result)
    rationale.extend(notes)

    resolution, notes = score_resolution(task, prior_actions, action, tool_result)
    rationale.extend(notes)

    clarity, notes = score_clarity(action)
    rationale.extend(notes)

    penalties, notes = score_penalties(task, prior_actions, action, tool_result)
    rationale.extend(notes)

    total = round(max(-1.0, min(1.0, tool_usage + reasoning + resolution + clarity - penalties)), 4)
    return Reward(
        total=total,
        tool_usage=round(tool_usage, 4),
        reasoning=round(reasoning, 4),
        resolution=round(resolution, 4),
        clarity=round(clarity, 4),
        penalties=round(penalties, 4),
        rationale=rationale,
    )


def grade_trajectory_components(task: TaskConfig, trajectory: list[dict[str, Any]]) -> dict[str, float]:
    prior_actions: list[Action] = []
    step_rewards: list[Reward] = []

    for entry in trajectory:
        action = entry["action"]
        tool_result = entry.get("tool_result")
        reward = compute_step_reward(task, prior_actions, action, tool_result)
        step_rewards.append(reward)
        prior_actions.append(action)

    if not step_rewards:
        return {"tool_usage": 0.0, "reasoning": 0.0, "resolution": 0.0, "response_quality": 0.0, "score": 0.0}

    required_tools = [item for item in task.expected_plan if item != "reply_to_user"]
    matched = 0.0
    cursor = 0
    tool_actions_taken = 0
    for action in prior_actions:
        if action.action_type == "reply_to_user":
            continue
        tool_actions_taken += 1
        if cursor < len(required_tools) and action.action_type == required_tools[cursor]:
            matched += 1.0
            cursor += 1
        elif action.action_type in required_tools[cursor + 1 :]:
            matched += 0.5

    tool_usage = 1.0 if not required_tools else max(0.0, min(1.0, matched / len(required_tools)))
    if tool_actions_taken > len(required_tools):
        tool_usage = max(0.0, tool_usage - 0.1 * (tool_actions_taken - len(required_tools)))

    reasoning = max(0.0, min(1.0, sum(item.reasoning for item in step_rewards) / (0.3 * len(step_rewards))))
    resolution = max(item.resolution for item in step_rewards) / 0.3
    response_quality = max(
        0.0,
        min(
            1.0,
            max(item.clarity for item in step_rewards) / 0.2,
        ),
    )

    score = round(
        (0.2 * tool_usage) + (0.3 * reasoning) + (0.3 * max(0.0, min(1.0, resolution))) + (0.2 * response_quality),
        4,
    )
    return {
        "tool_usage": round(max(0.0, min(1.0, tool_usage)), 4),
        "reasoning": round(reasoning, 4),
        "resolution": round(max(0.0, min(1.0, resolution)), 4),
        "response_quality": round(response_quality, 4),
        "score": score,
    }
