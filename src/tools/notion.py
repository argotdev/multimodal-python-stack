"""Notion run-sheet tool for logging and tracking."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import httpx

from src.core.types import ToolResult
from src.tools.base import Tool


class NotionRunSheetTool(Tool):
    """Update a Notion database (run-sheet style).

    This tool adds or updates entries in a Notion database,
    perfect for logging events, tracking tasks, or maintaining run-sheets.

    Example:
        tool = NotionRunSheetTool(
            api_key="secret_...",
            database_id="abc123..."
        )
        result = await tool.execute(
            title="Quality Check #42",
            status="completed",
            notes="All items passed inspection"
        )
    """

    name = "update_notion_runsheet"
    description = "Add or update an entry in the Notion run-sheet database"

    parameters = {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title/name of the entry",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "blocked"],
                "description": "Current status of the entry",
            },
            "notes": {
                "type": "string",
                "description": "Additional notes or observations",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags for categorization",
            },
        },
        "required": ["title", "status"],
    }

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    # Status emoji mapping
    STATUS_EMOJI = {
        "pending": "⏳",
        "in_progress": "🔄",
        "completed": "✅",
        "blocked": "🚫",
    }

    def __init__(
        self,
        api_key: str | None = None,
        database_id: str | None = None,
    ):
        """Initialize Notion run-sheet tool.

        Args:
            api_key: Notion API key (uses env var if not provided)
            database_id: Target database ID (uses env var if not provided)
        """
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.database_id = database_id or os.getenv("NOTION_DATABASE_ID")

        if not self.api_key:
            raise ValueError(
                "Notion API key required. Set NOTION_API_KEY env var "
                "or pass api_key parameter."
            )
        if not self.database_id:
            raise ValueError(
                "Notion database ID required. Set NOTION_DATABASE_ID env var "
                "or pass database_id parameter."
            )

    async def execute(
        self,
        title: str,
        status: str = "pending",
        notes: str = "",
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Add an entry to the Notion database.

        Args:
            title: Entry title
            status: Status (pending, in_progress, completed, blocked)
            notes: Additional notes
            tags: Optional list of tags

        Returns:
            ToolResult with page info or error
        """
        if status not in self.STATUS_EMOJI:
            status = "pending"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": self.NOTION_VERSION,
        }

        # Build page properties
        # Note: This assumes a standard database schema
        # Adjust property names to match your database
        properties = {
            "Title": {
                "title": [{"text": {"content": f"{self.STATUS_EMOJI[status]} {title}"}}]
            },
            "Status": {"select": {"name": status.replace("_", " ").title()}},
            "Timestamp": {"date": {"start": datetime.now().isoformat()}},
        }

        if notes:
            properties["Notes"] = {
                "rich_text": [{"text": {"content": notes}}]
            }

        if tags:
            properties["Tags"] = {
                "multi_select": [{"name": tag} for tag in tags]
            }

        payload = {
            "parent": {"database_id": self.database_id},
            "properties": properties,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/pages",
                    headers=headers,
                    json=payload,
                )

            if response.status_code in (200, 201):
                data = response.json()
                return ToolResult(
                    output={
                        "created": True,
                        "page_id": data.get("id"),
                        "url": data.get("url"),
                        "title": title,
                        "status": status,
                    }
                )
            else:
                error_data = response.json()
                return ToolResult(
                    error=f"Notion API error: {error_data.get('message', response.text)}"
                )

        except httpx.TimeoutException:
            return ToolResult(error="Notion request timed out")
        except httpx.HTTPError as e:
            return ToolResult(error=f"HTTP error: {e}")


class NotionQueryTool(Tool):
    """Query entries from a Notion database.

    Useful for retrieving recent entries or finding specific items.
    """

    name = "query_notion_runsheet"
    description = "Query entries from the Notion run-sheet database"

    parameters = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "blocked"],
                "description": "Filter by status (optional)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of entries to return (default 10)",
            },
        },
        "required": [],
    }

    BASE_URL = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"

    def __init__(
        self,
        api_key: str | None = None,
        database_id: str | None = None,
    ):
        self.api_key = api_key or os.getenv("NOTION_API_KEY")
        self.database_id = database_id or os.getenv("NOTION_DATABASE_ID")

    async def execute(
        self,
        status: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        """Query the Notion database.

        Args:
            status: Optional status filter
            limit: Maximum entries to return

        Returns:
            ToolResult with list of entries
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Notion-Version": self.NOTION_VERSION,
        }

        payload: dict[str, Any] = {
            "page_size": min(limit, 100),
            "sorts": [{"timestamp": "created_time", "direction": "descending"}],
        }

        if status:
            payload["filter"] = {
                "property": "Status",
                "select": {"equals": status.replace("_", " ").title()},
            }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.BASE_URL}/databases/{self.database_id}/query",
                    headers=headers,
                    json=payload,
                )

            if response.status_code == 200:
                data = response.json()
                entries = []
                for result in data.get("results", []):
                    props = result.get("properties", {})
                    entries.append({
                        "id": result.get("id"),
                        "title": self._extract_title(props.get("Title", {})),
                        "status": self._extract_select(props.get("Status", {})),
                        "created": result.get("created_time"),
                    })
                return ToolResult(output={"entries": entries, "count": len(entries)})
            else:
                return ToolResult(error=f"Notion API error: {response.text}")

        except httpx.HTTPError as e:
            return ToolResult(error=f"HTTP error: {e}")

    def _extract_title(self, prop: dict) -> str:
        title_list = prop.get("title", [])
        if title_list:
            return title_list[0].get("text", {}).get("content", "")
        return ""

    def _extract_select(self, prop: dict) -> str:
        select = prop.get("select", {})
        return select.get("name", "") if select else ""
