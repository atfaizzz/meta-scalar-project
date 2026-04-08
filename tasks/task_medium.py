from __future__ import annotations

from env.models import CustomerMetadata, TaskConfig


MEDIUM_TASK = TaskConfig(
    task_id="medium_order_delay",
    difficulty="medium",
    title="Order delay investigation",
    description="Investigate a delayed order, ground the answer in the order database, and explain the updated ETA.",
    ticket_id="TICK-ORD-440",
    initial_user_message=(
        "my hardware key still isnt here and onboarding is blocked. tracker hasnt moved since tuesday. "
        "order ORD-1001 was due yesterday. what is going on?"
    ),
    customer_metadata=CustomerMetadata(
        customer_id="CUST-002",
        name="Nikhil Rao",
        company="FinStack Labs",
        plan="Business",
        tier="priority",
        locale="en-IN",
        tenure_months=26,
        sentiment="frustrated",
        risk_level="medium",
        last_order_id="ORD-1001",
    ),
    available_tools=[
        "reply_to_user",
        "query_kb",
        "check_order_status",
        "issue_refund",
        "escalate",
    ],
    expected_plan=["check_order_status", "reply_to_user"],
    reasoning_markers=["order", "delay", "eta", "tracking"],
    resolution_keywords=["ord-1001", "weather", "2026-04-08", "delay"],
    order_id="ORD-1001",
    max_steps=5,
)


EXPECTED_OUTPUT = {
    "tool_sequence": ["check_order_status", "reply_to_user"],
    "required_keywords": ["ORD-1001", "weather", "2026-04-08", "delay"],
}


def grade(trajectory: list[dict]) -> float:
    from env.models import Action, ToolResult
    from env.reward import grade_trajectory_components

    normalized = []
    for step in trajectory:
        action = step["action"] if isinstance(step["action"], Action) else Action.model_validate(step["action"])
        tool_result = step.get("tool_result")
        if tool_result is not None and not isinstance(tool_result, ToolResult):
            tool_result = ToolResult.model_validate(tool_result)
        normalized.append({"action": action, "tool_result": tool_result})
    return grade_trajectory_components(MEDIUM_TASK, normalized)["score"]
