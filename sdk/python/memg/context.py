"""Context builder: merges conscious, recalled, and summary facts into a prompt string."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import List, Optional

from .types import Artifact, ConsciousFact, RecalledFact, RecalledSummary, TurnSummary

_DEDUP_PREFIXES = ("the user ", "user ", "user's ", "[historical] ")


def _normalize_for_dedup(s: str) -> str:
    s = s.lower().strip()
    for prefix in _DEDUP_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip()


def _estimate_tokens(s: str) -> int:
    words = len(s.split())
    if words > 0:
        return math.ceil(words * 1.3)
    # No spaces (likely CJK) — count Unicode code points, not bytes.
    runes = sum(1 for _ in s)
    if runes == 0:
        return 0
    return math.ceil(runes * 2 / 3)


def build_context(
    conscious_facts: Optional[List[ConsciousFact]] = None,
    recalled_facts: Optional[List[RecalledFact]] = None,
    summaries: Optional[List[RecalledSummary]] = None,
    turn_summaries: Optional[List[TurnSummary]] = None,
    artifacts: Optional[List[Artifact]] = None,
    total_token_budget: int = 4000,
    summary_token_budget: int = 1000,
) -> str:
    """Build a context string from memory components.

    Priority: conscious > turn_summaries > recalled > artifacts > summaries.
    Deduplicates by normalized content across components.
    """
    budget = total_token_budget if total_token_budget > 0 else 4000
    summary_budget = summary_token_budget if summary_token_budget > 0 else 1000

    parts: list = []
    seen: set = set()
    tokens_used = 0

    # 1. Conscious facts (highest priority)
    if conscious_facts:
        section_lines = ["User profile:"]
        for f in conscious_facts:
            normalized = _normalize_for_dedup(f.content)
            if normalized in seen:
                continue
            seen.add(normalized)
            line = f"- {f.content}"
            est = _estimate_tokens(line + "\n")
            if tokens_used + est > budget:
                break
            section_lines.append(line)
            tokens_used += est
        if len(section_lines) > 1:
            parts.append("\n".join(section_lines))

    # 2. Session context (turn summaries)
    if turn_summaries:
        section_lines = ["\nSession context:"]
        header_tokens = _estimate_tokens(section_lines[0])
        section_tokens = header_tokens
        any_added = False

        for ts in turn_summaries:
            if ts.is_overview:
                line = f"- [Overview] {ts.summary}"
            else:
                line = f"- [Turns {ts.start_turn}-{ts.end_turn}] {ts.summary}"
            est = _estimate_tokens(line)
            if tokens_used + section_tokens + est > budget:
                break
            section_lines.append(line)
            section_tokens += est
            any_added = True

        if any_added:
            parts.append("\n".join(section_lines))
            tokens_used += section_tokens

    # 3. Recalled facts (medium priority) - dedup against conscious
    if recalled_facts:
        section_lines = ["\nRelevant context from memory:"]
        header_tokens = _estimate_tokens(section_lines[0])
        section_tokens = header_tokens
        any_added = False

        for f in recalled_facts:
            normalized = _normalize_for_dedup(f.content)
            if normalized in seen:
                continue
            seen.add(normalized)
            prefix = "- "
            if f.temporal_status == "historical":
                prefix = "- [historical] "
            line = prefix + f.content
            est = _estimate_tokens(line)
            if tokens_used + section_tokens + est > budget:
                break
            section_lines.append(line)
            section_tokens += est
            any_added = True

        if any_added:
            parts.append("\n".join(section_lines))
            tokens_used += section_tokens

    # 4. Relevant code/data (artifacts)
    if artifacts:
        section_lines = ["\nRelevant code/data:"]
        header_tokens = _estimate_tokens(section_lines[0])
        section_tokens = header_tokens
        any_added = False

        for a in artifacts:
            entry_parts = []
            label = f"[{a.artifact_type}"
            if a.language:
                label += f":{a.language}"
            label += f"] {a.description}"
            entry_parts.append(f"- {label}")
            if a.artifact_type == "code" and a.language:
                entry_parts.append(f"  ```{a.language}")
                entry_parts.append(f"  {a.content}")
                entry_parts.append("  ```")
            else:
                entry_parts.append(f"  {a.content}")
            entry = "\n".join(entry_parts)
            est = _estimate_tokens(entry)
            if tokens_used + section_tokens + est > budget:
                break
            section_lines.append(entry)
            section_tokens += est
            any_added = True

        if any_added:
            parts.append("\n".join(section_lines))
            tokens_used += section_tokens

    # 5. Summaries (lowest priority, own sub-budget)
    if summaries:
        effective_budget = summary_budget
        remaining = budget - tokens_used
        if effective_budget > remaining:
            effective_budget = remaining

        if effective_budget > 0:
            section_lines = ["\nRelevant past conversations:"]
            header_tokens = _estimate_tokens(section_lines[0])
            section_tokens = header_tokens
            any_added = False

            for s in summaries:
                normalized = _normalize_for_dedup(s.summary)
                if normalized in seen:
                    continue
                seen.add(normalized)
                date_str = ""
                if s.created_at:
                    date_str = s.created_at.strftime("%b {}, %Y").format(s.created_at.day)
                line = f"- [{date_str}] {s.summary}" if date_str else f"- {s.summary}"
                est = _estimate_tokens(line)
                if section_tokens + est > effective_budget:
                    break
                section_lines.append(line)
                section_tokens += est
                any_added = True

            if any_added:
                parts.append("\n".join(section_lines))

    result = "\n".join(parts)
    return result.rstrip("\n")


def adaptive_window_size(total_budget: int, used_tokens: int, default_turns: int) -> int:
    """Calculate how many recent messages to load based on remaining token budget.

    Args:
        total_budget: Overall token budget.
        used_tokens: Tokens already consumed by other context components.
        default_turns: Default/maximum number of turns.

    Returns:
        Number of messages to include in the working memory window.
    """
    remaining = total_budget - used_tokens
    floor_tokens = 800
    if remaining < floor_tokens:
        remaining = floor_tokens
    avg_tokens_per_msg = 200
    max_msgs = remaining // avg_tokens_per_msg
    if max_msgs > default_turns:
        max_msgs = default_turns
    min_msgs = floor_tokens // avg_tokens_per_msg
    if max_msgs < min_msgs:
        max_msgs = min_msgs
    return max_msgs
