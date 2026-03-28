"""Artifact detection: scans messages for code blocks, JSON objects, SQL statements."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List


@dataclass
class DetectedArtifact:
    content: str
    artifact_type: str   # "code", "json", "sql"
    language: str
    source_role: str     # "user" or "assistant"


_CODE_FENCE_RE = re.compile(r"```(\w*)\n([\s\S]*?)```")
_SQL_PREFIX_RE = re.compile(r"(?i)^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP)\b")
_JSON_BLOCK_RE = re.compile(r"(?s)\{[\s\S]*?\}")


def detect_artifacts(messages: List[dict]) -> List[DetectedArtifact]:
    """Scan messages for code blocks, JSON objects, SQL statements.

    messages: list of dicts with 'role' and 'content' keys.
    Returns deduplicated list of DetectedArtifact.
    """
    seen: set = set()
    result: List[DetectedArtifact] = []

    def _add(a: DetectedArtifact) -> None:
        if a.content in seen:
            return
        seen.add(a.content)
        result.append(a)

    for msg in messages:
        if msg is None:
            continue
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        if not content:
            continue

        # 1. Code fences.
        for match in _CODE_FENCE_RE.finditer(content):
            lang = match.group(1).strip()
            body = match.group(2).strip()
            if not body:
                continue
            _add(DetectedArtifact(
                content=body,
                artifact_type="code",
                language=lang,
                source_role=role,
            ))

        # 2. Inline code blocks: consecutive lines starting with 4+ spaces or tab, > 3 lines.
        stripped = _CODE_FENCE_RE.sub("", content)
        lines = stripped.split("\n")
        block: List[str] = []
        for line in lines:
            if len(line) > 0 and (line.startswith("    ") or line[0] == "\t"):
                block.append(line)
            else:
                if len(block) > 3:
                    body = "\n".join(block).strip()
                    if body:
                        _add(DetectedArtifact(
                            content=body,
                            artifact_type="code",
                            language="",
                            source_role=role,
                        ))
                block = []
        if len(block) > 3:
            body = "\n".join(block).strip()
            if body:
                _add(DetectedArtifact(
                    content=body,
                    artifact_type="code",
                    language="",
                    source_role=role,
                ))

        # 3. JSON objects.
        for match in _JSON_BLOCK_RE.finditer(stripped):
            trimmed = match.group(0).strip()
            try:
                json.loads(trimmed)
            except (json.JSONDecodeError, ValueError):
                continue
            _add(DetectedArtifact(
                content=trimmed,
                artifact_type="json",
                language="",
                source_role=role,
            ))

        # 4. SQL statements.
        for line in lines:
            trimmed = line.strip()
            if _SQL_PREFIX_RE.match(trimmed):
                _add(DetectedArtifact(
                    content=trimmed,
                    artifact_type="sql",
                    language="",
                    source_role=role,
                ))

    return result
