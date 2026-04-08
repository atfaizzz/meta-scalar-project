from __future__ import annotations

from env.models import CustomerMetadata, TaskConfig


EASY_TASK = TaskConfig(
    task_id="easy_faq_plan_switch",
    difficulty="easy",
    title="FAQ: billing plan switch",
    description="Answer a plan-switch question using the knowledge base and close the ticket in one clean response.",
    ticket_id="TICK-FAQ-301",
    initial_user_message=(
        "hey team, quick q: if i move from monthly to annual later, do i keep all my dashboards "
        "+ old usage data or does anything reset?"
    ),
    customer_metadata=CustomerMetadata(
        customer_id="CUST-001",
        name="Ava Johnson",
        company="Northwind Studio",
        plan="Pro Monthly",
        tier="growth",
        locale="en-US",
        tenure_months=14,
        sentiment="curious",
        risk_level="low",
        last_order_id=None,
    ),
    available_tools=[
        "reply_to_user",
        "query_kb",
        "check_order_status",
        "issue_refund",
        "escalate",
    ],
    expected_plan=["query_kb", "reply_to_user"],
    reasoning_markers=["annual", "data", "billing"],
    resolution_keywords=["annual", "data", "dashboards", "prorated"],
    preferred_kb_article="billing_plan_switch",
    max_steps=4,
)


EXPECTED_OUTPUT = {
    "tool_sequence": ["query_kb", "reply_to_user"],
    "required_keywords": ["annual", "data", "dashboards", "prorated"],
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
    return grade_trajectory_components(EASY_TASK, normalized)["score"]
