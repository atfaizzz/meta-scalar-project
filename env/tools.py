from __future__ import annotations

import re
from typing import Any

from env.models import ToolResult


KNOWLEDGE_BASE: dict[str, dict[str, Any]] = {
    "billing_plan_switch": {
        "title": "Switching from monthly to annual billing",
        "body": (
            "Customers can switch from monthly to annual billing from Settings > Billing. "
            "Dashboards, workspaces, and historical usage data remain intact. The billing "
            "change is prorated on the next invoice and does not reset analytics history."
        ),
        "keywords": {"monthly", "annual", "billing", "data", "dashboard", "switch", "prorated"},
    },
    "shipping_delay_status": {
        "title": "Reading delayed shipment statuses",
        "body": (
            "If tracking shows a carrier or weather hold, agents should confirm the latest "
            "scan, share the updated ETA, and avoid promising manual refunds unless the "
            "shipment is lost or damaged."
        ),
        "keywords": {"shipping", "delay", "carrier", "weather", "eta", "tracking", "order"},
    },
    "refund_policy_enterprise": {
        "title": "Enterprise refund exceptions",
        "body": (
            "Standard refunds are available within 30 days of purchase. After 30 days, support "
            "agents cannot issue a full refund without finance manager approval. Verified "
            "platform outages or failed onboarding can be escalated for manual exception review."
        ),
        "keywords": {
            "refund",
            "policy",
            "30-day",
            "30",
            "manager",
            "exception",
            "outage",
            "onboarding",
            "enterprise",
        },
    },
}


ORDER_DATABASE: dict[str, dict[str, Any]] = {
    "ORD-1001": {
        "customer_id": "CUST-002",
        "status": "in_transit_delayed",
        "carrier": "BlueDart",
        "latest_scan": "Weather hold at Chennai hub",
        "eta": "2026-04-08",
        "shipped_at": "2026-04-01",
        "delay_days": 4,
        "items": ["Hardware security key"],
        "total": 89.0,
        "refund_window_days": 30,
    },
    "ORD-9007": {
        "customer_id": "CUST-003",
        "status": "delivered",
        "carrier": "DHL Express",
        "latest_scan": "Delivered to office front desk",
        "eta": "2026-02-10",
        "delivered_at": "2026-02-10",
        "purchased_at": "2026-02-01",
        "days_since_purchase": 64,
        "items": ["25 Analytics seats"],
        "total": 2499.0,
        "refund_window_days": 30,
        "incident_note": "Provisioning incident impacted the first 72 hours after purchase.",
    },
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9-]+", text.lower()))


class SupportTools:
    def list_available_tools(self) -> list[str]:
        return [
            "reply_to_user",
            "query_kb",
            "check_order_status",
            "issue_refund",
            "escalate",
        ]

    def query_kb(self, query: str) -> ToolResult:
        query_tokens = _tokenize(query)
        best_article = None
        best_score = -1
        for article_id, article in KNOWLEDGE_BASE.items():
            overlap = len(query_tokens & article["keywords"])
            if overlap > best_score:
                best_article = article_id
                best_score = overlap

        assert best_article is not None
        article = KNOWLEDGE_BASE[best_article]
        return ToolResult(
            tool="query_kb",
            success=True,
            summary=f"{article['title']}: {article['body']}",
            data={
                "article_id": best_article,
                "title": article["title"],
                "body": article["body"],
            },
        )

    def check_order_status(self, order_id: str) -> ToolResult:
        order = ORDER_DATABASE.get(order_id)
        if order is None:
            return ToolResult(
                tool="check_order_status",
                success=False,
                summary=f"Order {order_id} was not found.",
                data={"order_id": order_id},
            )

        if order["status"] == "in_transit_delayed":
            summary = (
                f"Order {order_id} is delayed in transit with {order['carrier']}. "
                f"Latest scan: {order['latest_scan']}. Updated ETA: {order['eta']}."
            )
        else:
            summary = (
                f"Order {order_id} was delivered on {order.get('delivered_at', order['eta'])}. "
                f"Latest scan: {order['latest_scan']}."
            )

        return ToolResult(
            tool="check_order_status",
            success=True,
            summary=summary,
            data={"order_id": order_id, **order},
        )

    def issue_refund(self, order_id: str, amount: float, reason: str) -> ToolResult:
        order = ORDER_DATABASE.get(order_id)
        if order is None:
            return ToolResult(
                tool="issue_refund",
                success=False,
                summary=f"Refund denied because order {order_id} does not exist.",
                data={"order_id": order_id, "approved": False},
            )

        if amount <= 0:
            return ToolResult(
                tool="issue_refund",
                success=False,
                summary="Refund denied because the amount must be greater than zero.",
                data={"order_id": order_id, "approved": False},
            )

        if order["status"] == "in_transit_delayed":
            approved_amount = min(amount, order["total"])
            return ToolResult(
                tool="issue_refund",
                success=True,
                summary=(
                    f"Refund approved for {order_id} in the amount of ${approved_amount:.2f} "
                    "because the shipment is materially delayed."
                ),
                data={
                    "order_id": order_id,
                    "approved": True,
                    "approved_amount": approved_amount,
                    "reason": reason,
                },
            )

        within_window = order.get("days_since_purchase", 0) <= order.get("refund_window_days", 30)
        if within_window:
            approved_amount = min(amount, order["total"])
            return ToolResult(
                tool="issue_refund",
                success=True,
                summary=f"Refund approved for {order_id} in the amount of ${approved_amount:.2f}.",
                data={
                    "order_id": order_id,
                    "approved": True,
                    "approved_amount": approved_amount,
                    "reason": reason,
                },
            )

        return ToolResult(
            tool="issue_refund",
            success=False,
            summary=(
                f"Refund denied for {order_id}: purchase is outside the 30-day policy window. "
                "Finance manager approval is required for any exception."
            ),
            data={
                "order_id": order_id,
                "approved": False,
                "requires_escalation": True,
                "policy_window_days": order.get("refund_window_days", 30),
                "days_since_purchase": order.get("days_since_purchase"),
                "reason": reason,
            },
        )
