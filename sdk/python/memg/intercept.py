from __future__ import annotations

import logging
import threading
from typing import Any, Optional, List

from .client import MemGClient

logger = logging.getLogger("memg")

_MEMORY_CONTEXT_HEADER = "[Memory Context]"
_MEMORY_CONTEXT_FOOTER = "[End Memory Context]"


def _format_memory_context(memories: list) -> str:
    """Format memories into a context block for injection."""
    lines = [
        _MEMORY_CONTEXT_HEADER,
        "The following are relevant memories about this user:",
    ]
    for m in memories:
        lines.append(f"- {m.content}")
    lines.append(_MEMORY_CONTEXT_FOOTER)
    return "\n".join(lines)


def _build_exchange_messages(
    request_messages: list, assistant_content: str
) -> List[dict]:
    """Build a role/content message list from request messages and assistant response."""
    exchange: List[dict] = []
    for msg in request_messages:
        role = None
        content = None
        if isinstance(msg, dict):
            role = msg.get("role")
            raw = msg.get("content", "")
            if isinstance(raw, str):
                content = raw
            elif isinstance(raw, list):
                texts = []
                for part in raw:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        texts.append(part)
                content = " ".join(texts)
        elif hasattr(msg, "role"):
            role = msg.role
            raw = msg.content if hasattr(msg, "content") else ""
            content = raw if isinstance(raw, str) else ""

        if role in ("user", "assistant") and content:
            exchange.append({"role": role, "content": content})

    if assistant_content:
        exchange.append({"role": "assistant", "content": assistant_content})

    return exchange


def _fire_extraction(
    mcp_client: MemGClient,
    entity: str,
    request_messages: list,
    assistant_content: str,
) -> None:
    """Send the full exchange to the server's extraction pipeline in a background thread.

    Falls back to raw add if the server doesn't support extract_from_messages.
    """
    exchange = _build_exchange_messages(request_messages, assistant_content)
    if not exchange:
        return

    def _run() -> None:
        try:
            mcp_client.extract_from_messages(entity, exchange)
        except Exception:
            # Fallback: server may not have extraction pipeline configured.
            logger.debug("memg: extract_from_messages unavailable, falling back to add")
            try:
                from .types import MemoryInput

                user_msg = _last_user_message_openai(request_messages)
                if user_msg:
                    mcp_client.add(entity, [MemoryInput(content=user_msg)])
            except Exception:
                logger.warning("memg: fallback extraction also failed", exc_info=True)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# OpenAI interception
# ---------------------------------------------------------------------------


def wrap_openai_client(
    client: Any,
    mcp_client: MemGClient,
    entity: str,
    extract: bool = True,
) -> Any:
    """Patch an OpenAI client's chat.completions.create to inject memory context.

    Wraps both sync and streaming paths. The original client object is mutated
    (its create method is replaced). Returns the same client for chaining.
    """
    original_create = client.chat.completions.create

    def _patched_create(*args: Any, **kwargs: Any) -> Any:
        messages = list(kwargs.get("messages") or (args[0] if args else []))
        stream = kwargs.get("stream", False)

        # 1. Find last user message for search query
        query = _last_user_message_openai(messages)

        # 2. Search for relevant memories (graceful degradation)
        augmented_messages = messages
        if query:
            result = mcp_client.search(entity, query)
            if result.memories:
                context_text = _format_memory_context(result.memories)
                augmented_messages = _inject_openai_context(
                    messages, context_text
                )

        # Replace messages in kwargs
        kwargs["messages"] = augmented_messages
        # Remove from positional args if present
        if args:
            args = args[1:]

        if stream:
            return _openai_stream_wrapper(
                original_create, mcp_client, entity, extract, messages, args, kwargs
            )

        # 3. Call original
        response = original_create(*args, **kwargs)

        # 4. Extract from full exchange in background
        if extract:
            try:
                content = response.choices[0].message.content
                if content:
                    _fire_extraction(mcp_client, entity, messages, content)
            except (IndexError, AttributeError):
                pass

        return response

    client.chat.completions.create = _patched_create
    return client


def _last_user_message_openai(messages: list) -> Optional[str]:
    """Extract the content of the last user message from OpenAI messages."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            # content could be a list of content parts
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        texts.append(part)
                return " ".join(texts) if texts else None
        elif hasattr(msg, "role") and msg.role == "user":
            content = msg.content
            if isinstance(content, str):
                return content
    return None


def _inject_openai_context(messages: list, context_text: str) -> list:
    """Inject memory context into OpenAI messages.

    If a system message exists, appends to it. Otherwise prepends a new one.
    """
    messages = [m.copy() if isinstance(m, dict) else m for m in messages]

    # Look for existing system message
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "system":
            existing = msg.get("content", "")
            messages[i] = {
                **msg,
                "content": existing + "\n\n" + context_text,
            }
            return messages

    # No system message found -- prepend one
    messages.insert(0, {"role": "system", "content": context_text})
    return messages


def _openai_stream_wrapper(
    original_create: Any,
    mcp_client: MemGClient,
    entity: str,
    extract: bool,
    request_messages: list,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Wrap an OpenAI streaming response to accumulate content for extraction."""
    stream = original_create(*args, **kwargs)

    class _StreamWrapper:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self._accumulated: list = []

        def __iter__(self):
            return self

        def __next__(self):
            try:
                chunk = next(self._inner)
            except StopIteration:
                if extract and self._accumulated:
                    full_content = "".join(self._accumulated)
                    _fire_extraction(mcp_client, entity, request_messages, full_content)
                raise

            # Accumulate content deltas
            try:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    self._accumulated.append(delta.content)
            except (IndexError, AttributeError):
                pass

            return chunk

        def __enter__(self):
            if hasattr(self._inner, "__enter__"):
                self._inner.__enter__()
            return self

        def __exit__(self, *exc_info):
            if extract and self._accumulated:
                full_content = "".join(self._accumulated)
                _fire_extraction(mcp_client, entity, request_messages, full_content)
            if hasattr(self._inner, "__exit__"):
                return self._inner.__exit__(*exc_info)
            return False

        def close(self):
            if extract and self._accumulated:
                full_content = "".join(self._accumulated)
                _fire_extraction(mcp_client, entity, request_messages, full_content)
                self._accumulated.clear()
            if hasattr(self._inner, "close"):
                self._inner.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    return _StreamWrapper(stream)


# ---------------------------------------------------------------------------
# Gemini interception
# ---------------------------------------------------------------------------


def wrap_gemini_client(
    client: Any,
    mcp_client: MemGClient,
    entity: str,
    extract: bool = True,
) -> Any:
    """Patch a Gemini GenerativeModel's generate_content to inject memory context.

    Returns the same client for chaining.
    """
    original_generate = client.generate_content

    def _patched_generate(*args: Any, **kwargs: Any) -> Any:
        prompt = args[0] if args else kwargs.get("contents", "")
        query = None
        if isinstance(prompt, str):
            query = prompt
        elif isinstance(prompt, list):
            for c in reversed(prompt):
                if isinstance(c, dict) and c.get("role") == "user":
                    parts = c.get("parts", [])
                    if parts and isinstance(parts[0], dict):
                        query = parts[0].get("text")
                        break

        # 1. Search for relevant memories
        if query:
            result = mcp_client.search(entity, query)
            if result.memories:
                context_text = _format_memory_context(result.memories)
                existing_sys = kwargs.get("system_instruction", getattr(client, "_system_instruction", None))
                if existing_sys:
                    kwargs["system_instruction"] = str(existing_sys) + "\n\n" + context_text
                else:
                    kwargs["system_instruction"] = context_text

        # 2. Call original
        response = original_generate(*args, **kwargs)

        # 3. Extract from full exchange in background
        if extract:
            try:
                resp_text = response.text if hasattr(response, "text") else ""
                if resp_text:
                    # Build exchange from Gemini contents
                    request_messages = _gemini_contents_to_messages(prompt)
                    _fire_extraction(mcp_client, entity, request_messages, resp_text)
            except Exception:
                pass

        return response

    client.generate_content = _patched_generate
    return client


def _gemini_contents_to_messages(prompt: Any) -> list:
    """Convert Gemini contents to a simple role/content message list."""
    messages: list = []
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        for c in prompt:
            if isinstance(c, dict):
                role = "assistant" if c.get("role") == "model" else c.get("role", "user")
                parts = c.get("parts", [])
                text = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in parts
                )
                if text.strip():
                    messages.append({"role": role, "content": text})
    return messages


# ---------------------------------------------------------------------------
# Anthropic interception
# ---------------------------------------------------------------------------


def wrap_anthropic_client(
    client: Any,
    mcp_client: MemGClient,
    entity: str,
    extract: bool = True,
) -> Any:
    """Patch an Anthropic client's messages.create to inject memory context.

    Anthropic uses `system` as a top-level parameter (not a message role).
    Wraps both sync and streaming paths. Returns the same client for chaining.
    """
    original_create = client.messages.create

    def _patched_create(*args: Any, **kwargs: Any) -> Any:
        messages = list(kwargs.get("messages", []))
        stream = kwargs.get("stream", False)

        # 1. Find last user message
        query = _last_user_message_anthropic(messages)

        # 2. Search for relevant memories (graceful degradation)
        if query:
            result = mcp_client.search(entity, query)
            if result.memories:
                context_text = _format_memory_context(result.memories)
                kwargs = _inject_anthropic_context(kwargs, context_text)

        if stream:
            return _anthropic_stream_wrapper(
                original_create, mcp_client, entity, extract, messages, args, kwargs
            )

        # 3. Call original
        response = original_create(*args, **kwargs)

        # 4. Extract from full exchange in background
        if extract:
            try:
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        _fire_extraction(mcp_client, entity, messages, block.text)
                        break
            except (IndexError, AttributeError):
                pass

        return response

    client.messages.create = _patched_create
    return client


def _last_user_message_anthropic(messages: list) -> Optional[str]:
    """Extract the content of the last user message from Anthropic messages."""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        texts.append(part)
                return " ".join(texts) if texts else None
    return None


def _inject_anthropic_context(kwargs: dict, context_text: str) -> dict:
    """Inject memory context into Anthropic kwargs.

    Anthropic's `system` is a top-level parameter, not a message role.
    """
    kwargs = dict(kwargs)
    existing_system = kwargs.get("system", "")

    if isinstance(existing_system, str) and existing_system:
        kwargs["system"] = existing_system + "\n\n" + context_text
    elif isinstance(existing_system, list):
        # system can be a list of content blocks
        kwargs["system"] = existing_system + [
            {"type": "text", "text": "\n\n" + context_text}
        ]
    else:
        kwargs["system"] = context_text

    return kwargs


def _anthropic_stream_wrapper(
    original_create: Any,
    mcp_client: MemGClient,
    entity: str,
    extract: bool,
    request_messages: list,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Wrap an Anthropic streaming response to accumulate content for extraction."""
    stream = original_create(*args, **kwargs)

    class _StreamWrapper:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self._accumulated: list = []

        def __iter__(self):
            return self

        def __next__(self):
            try:
                event = next(self._inner)
            except StopIteration:
                if extract and self._accumulated:
                    full_content = "".join(self._accumulated)
                    _fire_extraction(mcp_client, entity, request_messages, full_content)
                raise

            # Accumulate text deltas
            if hasattr(event, "type"):
                if event.type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta and hasattr(delta, "text") and delta.text:
                        self._accumulated.append(delta.text)

            return event

        def __enter__(self):
            if hasattr(self._inner, "__enter__"):
                self._inner.__enter__()
            return self

        def __exit__(self, *exc_info):
            if extract and self._accumulated:
                full_content = "".join(self._accumulated)
                _fire_extraction(mcp_client, entity, request_messages, full_content)
            if hasattr(self._inner, "__exit__"):
                return self._inner.__exit__(*exc_info)
            return False

        def close(self):
            if extract and self._accumulated:
                full_content = "".join(self._accumulated)
                _fire_extraction(mcp_client, entity, request_messages, full_content)
                self._accumulated.clear()
            if hasattr(self._inner, "close"):
                self._inner.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    return _StreamWrapper(stream)
