from __future__ import annotations

from env.models import CustomerMetadata, TaskConfig


HARD_TASK = TaskConfig(
    task_id="hard_refund_exception",
    difficulty="hard",
    title="Refund dispute beyond policy",
    description=(
        "Handle a high-stakes refund dispute that is outside policy, verify the case details, "
        "attempt the refund system, then escalate for manual exception review."
    ),
    ticket_id="TICK-RFD-992",
    initial_user_message=(
        "i'm beyond frustrated. we bought 25 analytics seats and setup was busted the first week. "
        "now i'm being told i'm 'outside the 30-day window' so nobody can help? i need a full refund sorted today."
    ),
    customer_metadata=CustomerMetadata(
        customer_id="CUST-003",
        name="Marta Silva",
        company="Atlas Commerce",
        plan="Enterprise Annual",
        tier="enterprise",
        locale="en-GB",
        tenure_months=38,
        sentiment="angry",
        risk_level="high",
        last_order_id="ORD-9007",
    ),
    available_tools=[
        "reply_to_user",
        "query_kb",
        "check_order_status",
        "issue_refund",
        "escalate",
    ],
    expected_plan=["check_order_status", "query_kb", "issue_refund", "escalate", "reply_to_user"],
    reasoning_markers=["30-day", "override", "outage", "escalate"],
    resolution_keywords=["30-day", "exception", "escalated", "review"],
    preferred_kb_article="refund_policy_enterprise",
    order_id="ORD-9007",
    refund_target_amount=2499.0,
    escalation_required=True,
    max_steps=6,
)


EXPECTED_OUTPUT = {
    "tool_sequence": ["check_order_status", "query_kb", "issue_refund", "escalate", "reply_to_user"],
    "required_keywords": ["30-day", "exception", "escalated", "review"],
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
    return grade_trajectory_components(HARD_TASK, normalized)["score"]
