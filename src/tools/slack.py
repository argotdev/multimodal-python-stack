"""Slack alert tool for sending notifications."""

from __future__ import annotations

import os
from typing import Any

import httpx

from src.core.types import ToolResult
from src.tools.base import Tool


class SlackAlertTool(Tool):
    """Send alerts to Slack via webhook.

    This tool sends formatted messages to a Slack channel using
    incoming webhooks. Supports severity levels and custom channels.

    Example:
        tool = SlackAlertTool(
            webhook_url="https://hooks.slack.com/services/...",
            default_channel="#alerts"
        )
        result = await tool.execute(
            message="Motion detected in Zone A",
            severity="warning"
        )
    """

    name = "send_slack_alert"
    description = "Send an alert message to a Slack channel"

    parameters = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The alert message to send",
            },
            "severity": {
                "type": "string",
                "enum": ["info", "warning", "critical"],
                "description": "Alert severity level (info, warning, critical)",
            },
            "channel": {
                "type": "string",
                "description": "Target Slack channel (optional, uses default if not specified)",
            },
        },
        "required": ["message", "severity"],
    }

    # Emoji mapping for severity levels
    SEVERITY_EMOJI = {
        "info": ":information_source:",
        "warning": ":warning:",
        "critical": ":rotating_light:",
    }

    # Color mapping for attachments
    SEVERITY_COLOR = {
        "info": "#36a64f",
        "warning": "#ffcc00",
        "critical": "#ff0000",
    }

    def __init__(
        self,
        webhook_url: str | None = None,
        default_channel: str = "#alerts",
        username: str = "Multimodal Agent",
        icon_emoji: str = ":robot_face:",
    ):
        """Initialize Slack alert tool.

        Args:
            webhook_url: Slack incoming webhook URL (uses env var if not provided)
            default_channel: Default channel for alerts
            username: Bot username shown in Slack
            icon_emoji: Bot icon emoji
        """
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.default_channel = default_channel
        self.username = username
        self.icon_emoji = icon_emoji

        if not self.webhook_url:
            raise ValueError(
                "Slack webhook URL required. Set SLACK_WEBHOOK_URL env var "
                "or pass webhook_url parameter."
            )

    async def execute(
        self,
        message: str,
        severity: str = "info",
        channel: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Send an alert to Slack.

        Args:
            message: Alert message text
            severity: Alert severity (info, warning, critical)
            channel: Target channel (optional)

        Returns:
            ToolResult with send status
        """
        if severity not in self.SEVERITY_EMOJI:
            severity = "info"

        emoji = self.SEVERITY_EMOJI[severity]
        color = self.SEVERITY_COLOR[severity]

        payload = {
            "channel": channel or self.default_channel,
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"{emoji} *[{severity.upper()}]* {message}",
                            },
                        },
                    ],
                },
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.webhook_url, json=payload)

            if response.status_code == 200:
                return ToolResult(
                    output={
                        "sent": True,
                        "channel": channel or self.default_channel,
                        "severity": severity,
                    }
                )
            else:
                return ToolResult(
                    error=f"Slack API error: {response.status_code} - {response.text}"
                )

        except httpx.TimeoutException:
            return ToolResult(error="Slack request timed out")
        except httpx.HTTPError as e:
            return ToolResult(error=f"HTTP error: {e}")
