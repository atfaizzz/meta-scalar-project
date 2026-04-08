from __future__ import annotations

from env.models import Action, ConversationTurn, EpisodeState, Observation, Reward, TaskConfig, ToolResult


class StateManager:
    def __init__(self) -> None:
        self._task: TaskConfig | None = None
        self._state: EpisodeState | None = None

    @property
    def task(self) -> TaskConfig:
        if self._task is None:
            raise RuntimeError("No task is active. Call reset() first.")
        return self._task

    @property
    def state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("No task is active. Call reset() first.")
        return self._state

    def start(self, task: TaskConfig) -> Observation:
        self._task = task
        self._state = EpisodeState(
            task_id=task.task_id,
            ticket_id=task.ticket_id,
            user_message=task.initial_user_message,
            conversation_history=[ConversationTurn(role="user", content=task.initial_user_message)],
            available_tools=task.available_tools,
            customer_metadata=task.customer_metadata,
            last_tool_result=None,
            action_trace=[],
            reward_trace=[],
            done=False,
            ticket_status="open",
            step_count=0,
            max_steps=task.max_steps,
        )
        return self.observation()

    def observation(self) -> Observation:
        current = self.state
        return Observation(
            ticket_id=current.ticket_id,
            task_id=current.task_id,
            user_message=current.user_message,
            conversation_history=current.conversation_history,
            available_tools=current.available_tools,
            customer_metadata=current.customer_metadata,
            last_tool_result=current.last_tool_result,
            step_count=current.step_count,
            max_steps=current.max_steps,
            done=current.done,
            ticket_status=current.ticket_status,
        )

    def apply_step(self, action: Action, reward: Reward, tool_result: ToolResult | None) -> None:
        current = self.state
        current.step_count += 1
        current.action_trace.append(action)
        current.reward_trace.append(reward)
        current.last_tool_result = tool_result

        if tool_result is not None:
            current.conversation_history.append(
                ConversationTurn(role="tool", content=f"{tool_result.tool}: {tool_result.summary}")
            )

        if action.action_type == "reply_to_user" and action.message:
            current.conversation_history.append(ConversationTurn(role="assistant", content=action.message))
        elif action.action_type == "escalate":
            note = action.escalation_note or "Escalated to senior support."
            current.conversation_history.append(ConversationTurn(role="assistant", content=f"[Escalation] {note}"))

    def finalize_if_complete(self, action: Action, resolution_score: float) -> bool:
        current = self.state
        task = self.task

        if current.step_count >= current.max_steps:
            current.done = True
            if current.ticket_status == "open":
                current.ticket_status = "unresolved"
            return True

        if action.action_type != "reply_to_user":
            return current.done

        if task.escalation_required:
            if self.has_action("escalate") and resolution_score >= 0.18:
                current.ticket_status = "escalated"
                current.done = True
        elif resolution_score >= 0.18:
            current.ticket_status = "resolved"
            current.done = True

        return current.done

    def has_action(self, action_type: str) -> bool:
        return any(action.action_type == action_type for action in self.state.action_trace)

    def action_trace(self) -> list[Action]:
        return list(self.state.action_trace)

    def reward_trace(self) -> list[Reward]:
        return list(self.state.reward_trace)

    def export(self) -> dict:
        current = self.state
        return {
            "task_id": current.task_id,
            "ticket_id": current.ticket_id,
            "user_message": current.user_message,
            "conversation_history": [turn.model_dump() for turn in current.conversation_history],
            "available_tools": current.available_tools,
            "customer_metadata": current.customer_metadata.model_dump(),
            "last_tool_result": None if current.last_tool_result is None else current.last_tool_result.model_dump(),
            "action_trace": [action.model_dump() for action in current.action_trace],
            "reward_trace": [reward.model_dump() for reward in current.reward_trace],
            "done": current.done,
            "ticket_status": current.ticket_status,
            "step_count": current.step_count,
            "max_steps": current.max_steps,
        }
