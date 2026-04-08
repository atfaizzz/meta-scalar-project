from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException

from env.models import Action, Observation, ResetRequest, Reward, ToolResult
from env.reward import compute_step_reward
from env.state_manager import StateManager
from env.tools import SupportTools
from tasks.task_easy import EASY_TASK
from tasks.task_hard import HARD_TASK
from tasks.task_medium import MEDIUM_TASK


TASK_REGISTRY = {
    EASY_TASK.task_id: EASY_TASK,
    MEDIUM_TASK.task_id: MEDIUM_TASK,
    HARD_TASK.task_id: HARD_TASK,
}


class CustomerSupportEnv:
    def __init__(self) -> None:
        self.tools = SupportTools()
        self.state_manager = StateManager()

    async def reset(self, task_id: str | None = None) -> Observation:
        selected_task_id = task_id or EASY_TASK.task_id
        if selected_task_id not in TASK_REGISTRY:
            raise KeyError(f"Unknown task_id: {selected_task_id}")
        return self.state_manager.start(TASK_REGISTRY[selected_task_id])

    async def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        task = self.state_manager.task
        prior_actions = self.state_manager.action_trace()
        tool_result = self._execute_action(task_id=task.task_id, action=action)
        reward = compute_step_reward(task, prior_actions, action, tool_result)
        self.state_manager.apply_step(action=action, reward=reward, tool_result=tool_result)
        done = self.state_manager.finalize_if_complete(action=action, resolution_score=reward.resolution)
        observation = self.state_manager.observation()
        info = {
            "ticket_status": self.state_manager.state.ticket_status,
            "reward_breakdown": reward.model_dump(),
            "last_tool_result": None if tool_result is None else tool_result.model_dump(),
        }
        return observation, reward, done, info

    def state(self) -> dict[str, Any]:
        return self.state_manager.export()

    def _execute_action(self, task_id: str, action: Action) -> ToolResult | None:
        task = TASK_REGISTRY[task_id]

        if action.action_type == "reply_to_user":
            return None
        if action.action_type == "query_kb":
            query = action.query or action.message or task.title
            return self.tools.query_kb(query)
        if action.action_type == "check_order_status":
            order_id = action.order_id or task.order_id or task.customer_metadata.last_order_id
            if not order_id:
                return ToolResult(
                    tool="check_order_status",
                    success=False,
                    summary="No order ID was available for this ticket.",
                    data={},
                )
            return self.tools.check_order_status(order_id)
        if action.action_type == "issue_refund":
            order_id = action.order_id or task.order_id or task.customer_metadata.last_order_id
            if not order_id:
                return ToolResult(
                    tool="issue_refund",
                    success=False,
                    summary="Refund denied because no order ID was provided.",
                    data={"approved": False},
                )
            amount = action.refund_amount if action.refund_amount is not None else (task.refund_target_amount or 0.0)
            reason = action.refund_reason or action.reasoning or "Customer requested a refund."
            return self.tools.issue_refund(order_id, amount, reason)
        if action.action_type == "escalate":
            note = action.escalation_note or "Escalated the case for manual review."
            return ToolResult(
                tool="escalate",
                success=True,
                summary=f"Escalation created: {note}",
                data={"queue": "senior_support", "note": note},
            )

        raise ValueError(f"Unsupported action: {action.action_type}")


env = CustomerSupportEnv()
app = FastAPI(title="CustomerSupportEnv", version="0.1.0")


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name": "CustomerSupportEnv",
        "tasks": list(TASK_REGISTRY.keys()),
        "endpoints": ["/reset", "/step", "/state"],
    }


@app.post("/reset")
async def reset_endpoint(payload: ResetRequest | None = None) -> dict[str, Any]:
    try:
        observation = await env.reset(None if payload is None else payload.task_id)
        return {"observation": observation.model_dump()}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/step")
async def step_endpoint(action: Action) -> dict[str, Any]:
    try:
        observation, reward, done, info = await env.step(action)
        return {
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
async def state_endpoint() -> dict[str, Any]:
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("env.environment:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
