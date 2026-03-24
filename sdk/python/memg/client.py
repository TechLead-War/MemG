from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, List

import httpx

from .types import AddResult, Memory, MemoryInput, SearchResult

logger = logging.getLogger("memg")


class MemGClient:
    """MCP JSON-RPC 2.0 client for the MemG server.

    Communicates with the MemG MCP server over HTTP POST to /mcp.
    """

    def __init__(self, mcp_url: str = "http://localhost:8686") -> None:
        self._url = mcp_url.rstrip("/") + "/mcp"
        self._http = httpx.Client(timeout=30.0)
        self._req_id = 0

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _call(self, tool_name: str, arguments: dict) -> dict:
        """Send a tools/call JSON-RPC request and return the parsed result."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        resp = self._http.post(self._url, json=payload)
        resp.raise_for_status()
        body = resp.json()

        if "error" in body and body["error"] is not None:
            raise MemGError(body["error"].get("message", "unknown JSON-RPC error"))

        result = body.get("result", {})

        if result.get("isError"):
            text = ""
            content = result.get("content", [])
            if content:
                text = content[0].get("text", "")
            raise MemGError(text or "tool call failed")

        content = result.get("content", [])
        if not content:
            return {}

        text = content[0].get("text", "{}")
        return json.loads(text)

    def add(self, entity_id: str, memories: List[MemoryInput]) -> AddResult:
        """Add memories for an entity."""
        mem_dicts = []
        for m in memories:
            d: dict = {"content": m.content}
            if m.type != "identity":
                d["type"] = m.type
            if m.significance != "medium":
                d["significance"] = m.significance
            if m.tag is not None:
                d["tag"] = m.tag
            mem_dicts.append(d)

        data = self._call("add_memories", {
            "entity_id": entity_id,
            "memories": mem_dicts,
        })
        return AddResult(
            inserted=data.get("inserted", 0),
            reinforced=data.get("reinforced", 0),
        )

    def search(self, entity_id: str, query: str, limit: int = 10) -> SearchResult:
        """Search memories for an entity using semantic hybrid search."""
        args: dict = {"entity_id": entity_id, "query": query}
        if limit != 10:
            args["limit"] = limit

        data = self._call("search_memories", args)
        return self._parse_search_result(data)

    def list(
        self,
        entity_id: str,
        limit: int = 50,
        type: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> SearchResult:
        """List memories for an entity, optionally filtered by type or tag."""
        args: dict = {"entity_id": entity_id}
        if limit != 50:
            args["limit"] = limit
        if type is not None:
            args["type"] = type
        if tag is not None:
            args["tag"] = tag

        data = self._call("list_memories", args)
        return self._parse_search_result(data)

    def delete(self, entity_id: str, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        data = self._call("delete_memory", {
            "entity_id": entity_id,
            "memory_id": memory_id,
        })
        return data.get("deleted", False)

    def delete_all(self, entity_id: str) -> int:
        """Delete all memories for an entity. Returns the number deleted."""
        data = self._call("delete_all_memories", {
            "entity_id": entity_id,
        })
        return data.get("deleted", 0)

    def extract_from_messages(
        self,
        entity_id: str,
        messages: List[dict],
    ) -> int:
        """Extract structured memories from conversation messages using the
        server's LLM-powered extraction pipeline.

        Requires the MCP server to be started with --llm-provider.

        Args:
            entity_id: External entity identifier.
            messages: List of dicts with 'role' and 'content' keys.

        Returns:
            Number of facts extracted and stored.
        """
        data = self._call("extract_from_messages", {
            "entity_id": entity_id,
            "messages": messages,
        })
        return data.get("extracted", 0)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def _parse_search_result(self, data: dict) -> SearchResult:
        """Parse a search/list response into a SearchResult."""
        raw_memories = data.get("memories", [])
        memories = []
        for m in raw_memories:
            created_at = None
            if "created_at" in m:
                try:
                    created_at = datetime.fromisoformat(
                        m["created_at"].replace("Z", "+00:00")
                    )
                except (ValueError, AttributeError):
                    pass

            memories.append(Memory(
                id=m.get("id", ""),
                content=m.get("content", ""),
                type=m.get("type", "identity"),
                temporal_status=m.get("temporal_status", "current"),
                significance=m.get("significance", "medium"),
                created_at=created_at,
                tag=m.get("tag"),
                score=m.get("score"),
                reinforced_count=m.get("reinforced_count", 0),
            ))

        return SearchResult(
            memories=memories,
            count=data.get("count", len(memories)),
        )


class MemGError(Exception):
    """Raised when the MemG MCP server returns an error."""
