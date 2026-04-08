from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ActionType = Literal[
    "reply_to_user",
    "query_kb",
    "check_order_status",
    "issue_refund",
    "escalate",
]


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class CustomerMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str
    name: str
    company: str
    plan: str
    tier: str
    locale: str
    tenure_months: int
    sentiment: str
    risk_level: Literal["low", "medium", "high"]
    last_order_id: str | None = None


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: str
    success: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    task_id: str
    user_message: str
    conversation_history: list[ConversationTurn]
    available_tools: list[ActionType]
    customer_metadata: CustomerMetadata
    last_tool_result: ToolResult | None = None
    step_count: int = 0
    max_steps: int = 6
    done: bool = False
    ticket_status: Literal["open", "resolved", "escalated", "unresolved"] = "open"


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    message: str | None = None
    query: str | None = None
    order_id: str | None = None
    refund_amount: float | None = None
    refund_reason: str | None = None
    escalation_note: str | None = None
    reasoning: str = ""


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float
    tool_usage: float = 0.0
    reasoning: float = 0.0
    resolution: float = 0.0
    clarity: float = 0.0
    penalties: float = 0.0
    rationale: list[str] = Field(default_factory=list)


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    description: str
    ticket_id: str
    initial_user_message: str
    customer_metadata: CustomerMetadata
    available_tools: list[ActionType]
    expected_plan: list[ActionType]
    reasoning_markers: list[str]
    resolution_keywords: list[str]
    preferred_kb_article: str | None = None
    order_id: str | None = None
    refund_target_amount: float | None = None
    escalation_required: bool = False
    max_steps: int = 6


class EpisodeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    ticket_id: str
    user_message: str
    conversation_history: list[ConversationTurn]
    available_tools: list[ActionType]
    customer_metadata: CustomerMetadata
    last_tool_result: ToolResult | None = None
    action_trace: list[Action] = Field(default_factory=list)
    reward_trace: list[Reward] = Field(default_factory=list)
    done: bool = False
    ticket_status: Literal["open", "resolved", "escalated", "unresolved"] = "open"
    step_count: int = 0
    max_steps: int = 6


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = None
