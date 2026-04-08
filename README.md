# CustomerSupportEnv

CustomerSupportEnv is a production-style OpenEnv project that simulates an autonomous SaaS customer support workflow. The agent must resolve realistic support tickets by combining customer communication with internal tool usage across a knowledge base, an order database, a refund system, and an escalation path.

## Real-World Motivation

Modern SaaS support agents rarely solve tickets from the user message alone. They need to ground answers in policy, verify transactional facts, and make safe escalation decisions when policy and customer impact conflict. This environment models that operational reality with deterministic backend systems, noisy user language, partial rewards, and trajectory-based grading.

## Environment API

The environment implements the required OpenEnv-style API:

- `async reset(task_id: str | None = None) -> Observation`
- `async step(action: Action) -> tuple[Observation, Reward, bool, dict]`
- `state() -> dict`

`step()` always returns `observation, reward, done, info`.

## Observation Space

Each `Observation` includes:

- `ticket_id`
- `task_id`
- `user_message`
- `conversation_history`
- `available_tools`
- `customer_metadata`
- `last_tool_result`
- `step_count`
- `max_steps`
- `done`
- `ticket_status`

## Action Space

The `Action` model supports:

- `reply_to_user`
- `query_kb`
- `check_order_status`
- `issue_refund`
- `escalate`

Optional action fields include `message`, `query`, `order_id`, `refund_amount`, `refund_reason`, `escalation_note`, and `reasoning`.

## Task Suite

The benchmark ships with three deterministic tasks:

1. Easy: FAQ answering about switching from monthly to annual billing without data loss.
2. Medium: Order delay investigation that requires checking the order database before replying.
3. Hard: Refund dispute outside policy that requires order verification, policy lookup, refund-system evaluation, and escalation for exception review.

The user inputs include realistic ambiguity, typos, and emotionally varied phrasing, but the backend data and grading remain deterministic for reproducibility.

## Reward Design

Dense rewards are computed at every step with the following components:

- `+0.2` correct tool usage
- `+0.3` correct intermediate reasoning
- `+0.3` correct final resolution
- `+0.2` response clarity

Penalties are applied for:

- hallucinated claims before tool confirmation
- invalid actions or missing required fields
- repeated or irrelevant steps

This produces non-binary reward traces that encourage grounded multi-step behavior rather than shortcut memorization.

## Grading

Trajectory-based grading lives in `tasks/grader.py` and scores:

- tool usage
- reasoning quality
- final resolution quality
- response quality

Each component is deterministic and reproducible, and the final task score is returned in `[0.0, 1.0]`.

## Project Layout

```text
.
├── env/
│   ├── environment.py
│   ├── models.py
│   ├── reward.py
│   ├── state_manager.py
│   └── tools.py
├── tasks/
│   ├── grader.py
│   ├── task_easy.py
│   ├── task_hard.py
│   └── task_medium.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Environment

Run the FastAPI server locally:

```bash
uvicorn env.environment:app --host 0.0.0.0 --port 8000
```

Core endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`

## Running Inference

Set the required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Then run:

```bash
python inference.py
```

The script runs all three tasks and logs only the required lines:

```text
[START] task=... env=... model=...
[STEP] step=... action=... reward=... done=... error=...
[END] success=... steps=... rewards=...
```

## Expected Baseline Scores

With the built-in fallback policy in `inference.py`, the environment should produce strong deterministic baseline grades:

- Easy: about `0.95`
- Medium: about `0.90`
- Hard: about `0.88`

Model-driven runs can exceed or underperform those baselines depending on tool discipline and reply quality.

## Resource Profile

The environment is lightweight and designed to fit within:

- `2 vCPU`
- `8 GB RAM`
- inference runtime well below `20 minutes`

## Hugging Face Spaces / Docker

The included `Dockerfile` uses a lightweight Python 3.11 slim image, installs only the required dependencies, and starts the FastAPI app directly, making it suitable for container-based OpenEnv deployment and Hugging Face Spaces-style hosting.
