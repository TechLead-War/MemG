from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from ..client import MemGClient
from ..intercept import wrap_anthropic_client
from ..proxy import wrap_anthropic_proxy

logger = logging.getLogger("memg")


def wrap(
    client: Any,
    entity: Optional[str] = None,
    mode: str = "native",
    proxy_url: str = "http://localhost:8787/v1",
    mcp_url: str = "http://localhost:8686",
    extract: bool = True,
    config: Any = None,
    embedder: Any = None,
    **kwargs: Any,
) -> Any:
    """Wrap an Anthropic client with MemG memory.

    Args:
        client: An Anthropic client instance.
        entity: Entity identifier for memory scoping.
        mode: "native" for in-process, "proxy" for Go proxy, "client" for MCP.
        proxy_url: MemG proxy URL (proxy mode only).
        mcp_url: MemG MCP server URL (client mode only).
        extract: Whether to extract memories from responses.
        config: MemGConfig for native mode.
        embedder: Custom Embedder instance for native mode.

    Returns:
        A wrapped client instance.
    """
    if mode == "proxy":
        return wrap_anthropic_proxy(client, entity, proxy_url)
    elif mode == "client":
        if entity is None:
            raise ValueError("entity is required for client mode")
        mcp = MemGClient(mcp_url)
        return wrap_anthropic_client(client, mcp, entity, extract)
    elif mode == "native":
        if entity is None:
            raise ValueError("entity is required for native mode")
        return _wrap_anthropic_native(client, entity, extract, config, embedder, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _wrap_anthropic_native(
    client: Any,
    entity: str,
    extract: bool,
    config: Any,
    embedder: Any,
    **kwargs: Any,
) -> Any:
    """Patch an Anthropic client for native in-process memory."""
    from .. import MemG, MemGConfig

    cfg = config or MemGConfig.from_kwargs(**kwargs)
    engine = MemG(config=cfg, embedder=embedder)
    engine._ensure_native()

    original_create = client.messages.create

    def _patched_create(*args: Any, **call_kwargs: Any) -> Any:
        messages = list(call_kwargs.get("messages", []))
        stream = call_kwargs.get("stream", False)

        from .. import _last_user_content
        from ..recall import recall_facts, recall_summaries
        from ..context import build_context
        from ..session import ensure_session, save_exchange, load_recent_history
        from ..extract import run_extraction

        entity_uuid = engine._resolve_entity(entity)
        process_id = kwargs.get("process_id", "default")

        session, _ = ensure_session(
            engine._store, entity_uuid, process_id, cfg.session_timeout
        )

        conscious = []
        if cfg.conscious_mode:
            conscious = engine._load_conscious_facts(entity_uuid)

        query = _last_user_content_anthropic(messages)
        recalled = []
        summaries_recalled = []

        if query:
            query_vecs = engine._embedder.embed([query])
            if not query_vecs:
                raise RuntimeError("memg: embed recall query returned no vectors")
            query_vec = query_vecs[0]
            recalled = recall_facts(
                engine._engine, engine._store,
                query_vec, query, entity_uuid,
                limit=cfg.recall_limit,
                threshold=cfg.recall_threshold,
                max_candidates=cfg.max_recall_candidates,
            )
            summaries_recalled = recall_summaries(
                engine._engine, engine._store,
                query_vec, query, entity_uuid,
                limit=5, threshold=cfg.recall_threshold,
            )
            if recalled:
                fact_ids = [r.id for r in recalled]
                try:
                    engine._store.update_recall_usage(fact_ids)
                except Exception:
                    pass

        context_str = build_context(
            conscious_facts=conscious,
            recalled_facts=recalled,
            summaries=summaries_recalled,
            total_token_budget=cfg.memory_token_budget,
            summary_token_budget=cfg.summary_token_budget,
        )

        if context_str:
            call_kwargs = _inject_anthropic_context(call_kwargs, context_str)

        if stream:
            return _anthropic_native_stream(
                original_create, engine, entity, entity_uuid,
                session, extract, cfg, messages, args, call_kwargs, kwargs,
            )

        response = original_create(*args, **call_kwargs)

        content = ""
        try:
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    content = block.text
                    break
        except (IndexError, AttributeError):
            pass

        user_msgs = [
            {"role": m.get("role", ""), "content": _extract_text_content(m)}
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
        ]
        save_exchange(engine._store, session.uuid, entity_uuid, user_msgs, content)

        if extract and content:
            def _bg():
                try:
                    llm_call = engine._make_extraction_llm_call(
                        cfg.llm_provider, cfg.llm_model, kwargs
                    )
                    user_only = [m for m in user_msgs if m.get("role") == "user"]
                    run_extraction(
                        engine._store, entity_uuid, user_only,
                        llm_call, engine._embedder,
                    )
                except Exception as e:
                    logger.warning("memg: extraction failed: %s", e)
            t = threading.Thread(target=_bg, daemon=True)
            t.start()

        return response

    client.messages.create = _patched_create
    client._memg_engine = engine
    return client


def _last_user_content_anthropic(messages: list) -> Optional[str]:
    """Extract last user message content from Anthropic messages."""
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


def _extract_text_content(msg: dict) -> str:
    """Extract text content from an Anthropic message dict."""
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
        return " ".join(texts)
    return str(content)


def _inject_anthropic_context(kwargs: dict, context_text: str) -> dict:
    """Inject memory context into Anthropic kwargs."""
    kwargs = dict(kwargs)
    existing_system = kwargs.get("system", "")

    if isinstance(existing_system, str) and existing_system:
        kwargs["system"] = existing_system + "\n\n" + context_text
    elif isinstance(existing_system, list):
        kwargs["system"] = existing_system + [
            {"type": "text", "text": "\n\n" + context_text}
        ]
    else:
        kwargs["system"] = context_text

    return kwargs


def _anthropic_native_stream(
    original_create,
    engine,
    entity,
    entity_uuid,
    session,
    extract,
    cfg,
    original_messages,
    args,
    kwargs,
    engine_kwargs,
):
    """Wrap Anthropic streaming with native memory extraction."""
    stream = original_create(*args, **kwargs)

    class _StreamWrapper:
        def __init__(self, inner):
            self._inner = inner
            self._accumulated = []

        def __iter__(self):
            return self

        def __next__(self):
            try:
                event = next(self._inner)
            except StopIteration:
                self._finalize()
                raise

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
            self._finalize()
            if hasattr(self._inner, "__exit__"):
                return self._inner.__exit__(*exc_info)
            return False

        def close(self):
            self._finalize()
            if hasattr(self._inner, "close"):
                self._inner.close()

        def _finalize(self):
            if not self._accumulated:
                return
            content = "".join(self._accumulated)
            self._accumulated.clear()

            from ..session import save_exchange
            from ..extract import run_extraction

            user_msgs = [
                {"role": m.get("role", ""), "content": _extract_text_content(m)}
                for m in original_messages
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ]
            save_exchange(engine._store, session.uuid, entity_uuid, user_msgs, content)

            if extract and content:
                def _bg():
                    try:
                        llm_call = engine._make_extraction_llm_call(
                            cfg.llm_provider, cfg.llm_model, engine_kwargs
                        )
                        user_only = [m for m in user_msgs if m.get("role") == "user"]
                        run_extraction(
                            engine._store, entity_uuid, user_only,
                            llm_call, engine._embedder,
                        )
                    except Exception as e:
                        logger.warning("memg: extraction failed: %s", e)
                t = threading.Thread(target=_bg, daemon=True)
                t.start()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _StreamWrapper(stream)
