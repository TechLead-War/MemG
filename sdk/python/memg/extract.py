"""Extraction pipeline: trivial detection, LLM extraction, validation, dedup."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from .search import cosine_similarity
from .store import Store
from .types import Fact, FactFilter

logger = logging.getLogger("memg")

TRIVIAL_PATTERNS = [
    "thanks", "thank you", "ok", "okay", "got it", "sure",
    "hello", "hi", "hey", "bye", "goodbye", "good morning",
    "good night", "good evening", "good afternoon",
    "cool", "great", "awesome", "nice",
    "understood", "roger", "ack", "k", "kk", "lol", "haha",
]

VALID_TAGS = {
    "skill", "preference", "relationship", "medical", "location",
    "work", "hobby", "personal", "financial", "other",
}

SEMANTIC_DEDUP_THRESHOLD = 0.92
PROMOTION_THRESHOLD = 5
SIGNIFICANCE_HIGH = 10
SIGNIFICANCE_MEDIUM = 5


def content_key(content: str) -> str:
    """Compute content key: sha256(lowercase, strip punct, collapse whitespace)[:16].

    First 8 bytes of the hash as hex = 16 character string.
    """
    lower = content.lower()
    cleaned = ""
    for ch in lower:
        if ch.isalpha() or ch.isdigit() or ch.isspace():
            cleaned += ch
    normalized = " ".join(cleaned.split())
    h = hashlib.sha256(normalized.encode("utf-8")).digest()
    return h[:8].hex()


def ttl_for_significance(significance: int) -> Optional[datetime]:
    """Compute expiry time based on significance level."""
    if significance >= SIGNIFICANCE_HIGH:
        return None
    now = datetime.now(timezone.utc)
    if significance >= SIGNIFICANCE_MEDIUM:
        return now + timedelta(days=30)
    return now + timedelta(days=7)


def is_trivial_turn(messages: List[dict]) -> bool:
    """Check if messages contain only trivial content."""
    if not messages:
        return True

    for msg in messages:
        role = msg.get("role", "")
        if role != "user":
            continue
        raw = msg.get("content", "")
        if not raw:
            continue
        cleaned = raw.lower().strip()
        cleaned = "".join(
            ch for ch in cleaned
            if ch.isalpha() or ch.isdigit() or ch == " "
        ).strip()
        if not cleaned:
            continue

        is_triv = any(cleaned == p for p in TRIVIAL_PATTERNS)
        if not is_triv:
            return False

    return True


def build_extraction_prompt() -> str:
    """Build the LLM extraction prompt with today's date."""
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

    return (
        f"You are a knowledge extraction engine. Today's date is {today}.\n\n"
        "Extract facts from this conversation. Return a JSON array. Each fact:\n"
        "{\n"
        '  "content": "the fact as a clear statement about the user",\n'
        '  "type": "identity|event|pattern",\n'
        '  "significance": 1-10,\n'
        '  "tag": "category label",\n'
        '  "slot": "semantic slot name (e.g. location, job, diet, name, email, relationship, preference)",\n'
        '  "reference_time": "ISO date if time-bound, empty string if not",\n'
        '  "confidence": 0.0-1.0\n'
        "}\n\n"
        "Rules:\n"
        '- "identity" = enduring truths (preferences, attributes, relationships)\n'
        '- "event" = things that happened at a specific time\n'
        '- "pattern" = behavioral tendencies observed across the conversation\n'
        '- "tag" = a category label. Use one of: skill, preference, relationship, medical, location, work, hobby, personal, financial, or other\n'
        '- "slot" = the semantic slot this fact fills. Use: location, job, diet, name, email, relationship, preference, medical, hobby, skill, or other\n'
        '- "reference_time" = ISO 8601 date (YYYY-MM-DD) for time-bound facts, empty string otherwise\n'
        '- "confidence" = how confident you are (1.0 = explicitly stated by user, 0.5 = inferred, 0.0 = guessing)\n'
        f'- Resolve relative dates: "today" -> "{today}", "yesterday" -> "{yesterday}"\n'
        "- Significance: 10 = life-critical (allergies, medical), 7-9 = important (job, location), 4-6 = moderate, 1-3 = trivial (lunch, weather)\n"
        "- Skip greetings, filler, \"thank you\", and trivial exchanges\n"
        "- If nothing is worth extracting, return []\n\n"
        "Return ONLY the JSON array, no other text."
    )


def parse_extraction_response(content: str) -> list:
    """Parse JSON array from LLM response, resilient to markdown wrappers."""
    content = content.strip()

    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        pass

    stripped = _strip_code_fences(content)
    if stripped != content:
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            pass

    start = content.find("[")
    end = content.rfind("]")
    if start >= 0 and end > start:
        candidate = content[start:end + 1]
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("memg extract: could not parse JSON from response: %.100s", content)
    return []


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        idx = s.find("\n")
        if idx >= 0:
            s = s[idx + 1:]
        else:
            s = s.lstrip("`").lstrip("json").lstrip()
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def validate_extraction(facts: list) -> list:
    """Filter and validate extracted facts."""
    valid = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        c = (f.get("content") or "").strip()
        if not c:
            continue
        if len(c) > 500:
            continue

        ref_time = f.get("reference_time", "")
        if ref_time:
            try:
                datetime.strptime(ref_time, "%Y-%m-%d")
            except ValueError:
                f["reference_time"] = ""

        confidence = f.get("confidence")
        if confidence is not None:
            try:
                confidence = float(confidence)
                confidence = max(0.0, min(1.0, confidence))
                f["confidence"] = confidence
            except (TypeError, ValueError):
                f["confidence"] = None

        tag = (f.get("tag") or "").lower().strip()
        if tag and tag not in VALID_TAGS:
            tag = "other"
        f["tag"] = tag

        f["content"] = c
        valid.append(f)
    return valid


def _resolve_fact_type(t: str) -> str:
    t = (t or "").lower().strip()
    if t in ("event", "pattern", "identity"):
        return t
    return "identity"


def _clamp_significance(v) -> int:
    try:
        v = int(v)
    except (TypeError, ValueError):
        return SIGNIFICANCE_MEDIUM
    if v < 1 or v > 10:
        return SIGNIFICANCE_MEDIUM
    return v


def _confidence_value(v) -> float:
    if v is None:
        return 0.8
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.8


def extracted_to_facts(
    extracted: list,
    embedding_model: str = "",
    embedder=None,
) -> List[Fact]:
    """Convert validated extraction dicts to Fact objects.

    Optionally embeds fact content using the provided embedder.
    """
    if not extracted:
        return []

    facts = []
    contents = []

    for ef in extracted:
        fact_type = _resolve_fact_type(ef.get("type", ""))
        significance = _clamp_significance(ef.get("significance", 5))

        f = Fact(
            uuid="",
            content=ef["content"],
            fact_type=fact_type,
            temporal_status="current",
            significance=significance,
            tag=(ef.get("tag") or "").lower().strip(),
            slot=(ef.get("slot") or "").lower().strip(),
            confidence=_confidence_value(ef.get("confidence")),
            embedding_model=embedding_model,
            source_role="user",
            content_key=content_key(ef["content"]),
            expires_at=ttl_for_significance(significance),
        )

        ref_time = ef.get("reference_time", "")
        if ref_time:
            try:
                f.reference_time = datetime.strptime(ref_time, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        facts.append(f)
        contents.append(ef["content"])

    if embedder and contents:
        try:
            embeddings = embedder.embed(contents)
            for i, emb in enumerate(embeddings):
                if i < len(facts):
                    facts[i].embedding = emb
        except Exception as e:
            logger.warning("memg extract: embedding failed: %s", e)

    return facts


def persist_facts(
    store: Store,
    entity_uuid: str,
    facts: List[Fact],
) -> tuple:
    """Persist facts with dedup, slot conflict resolution, and promotion.

    Returns (inserted_count, reinforced_count).
    """
    if not facts or not entity_uuid:
        return 0, 0

    existing_facts = store.list_facts_for_recall(
        entity_uuid,
        FactFilter(statuses=["current"], exclude_expired=True),
        limit=500,
    )

    inserted = 0
    reinforced = 0

    for f in facts:
        if not f.content_key:
            f.content_key = content_key(f.content)

        if f.expires_at is None and f.significance < SIGNIFICANCE_HIGH:
            f.expires_at = ttl_for_significance(f.significance)

        # Step 1: Exact dedup by content key
        existing = store.find_fact_by_key(entity_uuid, f.content_key)
        if existing is not None:
            _reinforce_and_promote(store, existing, f.expires_at)
            reinforced += 1
            continue

        # Step 2: Slot conflict resolution for mutable identity facts
        if f.slot and f.fact_type == "identity" and f.temporal_status == "current":
            for ex in existing_facts:
                if (ex.slot == f.slot and
                        ex.temporal_status == "current" and
                        ex.fact_type == "identity"):
                    store.update_temporal_status(ex.uuid, "historical")
                    ex.temporal_status = "historical"

        # Step 3: Semantic dedup by embedding similarity
        if f.embedding and existing_facts:
            similar = _find_semantic_match(f.embedding, existing_facts)
            if similar is not None:
                _reinforce_and_promote(store, similar, f.expires_at)
                reinforced += 1
                continue

        # Step 4: Insert new fact
        store.insert_fact(entity_uuid, f)
        inserted += 1

    return inserted, reinforced


def _reinforce_and_promote(
    store: Store,
    existing: Fact,
    new_expires_at: Optional[datetime],
) -> None:
    """Reinforce an existing fact and promote if threshold is crossed."""
    new_count = existing.reinforced_count + 1
    should_promote = new_count >= PROMOTION_THRESHOLD and existing.significance < SIGNIFICANCE_HIGH

    expires_at = new_expires_at
    if should_promote:
        expires_at = None

    store.reinforce_fact(existing.uuid, expires_at)

    if should_promote:
        store.update_significance(existing.uuid, SIGNIFICANCE_HIGH)
        logger.info(
            "memg extract: promoted fact %s to high significance (reinforced %d times)",
            existing.uuid, new_count,
        )


def _find_semantic_match(
    embedding: List[float],
    candidates: List[Fact],
) -> Optional[Fact]:
    """Find the most similar existing fact by embedding, or None."""
    best = None
    best_score = 0.0

    for f in candidates:
        if not f.embedding:
            continue
        score = cosine_similarity(embedding, f.embedding)
        if score > best_score:
            best_score = score
            best = f

    if best_score >= SEMANTIC_DEDUP_THRESHOLD:
        return best
    return None


def run_extraction(
    store: Store,
    entity_uuid: str,
    messages: List[dict],
    llm_call,
    embedder=None,
) -> tuple:
    """Full extraction pipeline.

    Args:
        store: SQLiteStore instance
        entity_uuid: entity UUID
        messages: list of message dicts with 'role' and 'content'
        llm_call: callable(system_prompt, user_content) -> response_text
        embedder: optional Embedder for fact embedding

    Returns (inserted, reinforced).
    """
    if not messages:
        return 0, 0

    if is_trivial_turn(messages):
        return 0, 0

    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return 0, 0

    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in user_messages
    )

    prompt = build_extraction_prompt()

    try:
        response_text = llm_call(prompt, transcript)
    except Exception as e:
        logger.warning("memg extract: LLM call failed: %s", e)
        return 0, 0

    raw_facts = parse_extraction_response(response_text)
    validated = validate_extraction(raw_facts)
    if not validated:
        return 0, 0

    embedding_model = embedder.model_name() if embedder else ""
    facts = extracted_to_facts(validated, embedding_model, embedder)
    if not facts:
        return 0, 0

    return persist_facts(store, entity_uuid, facts)
