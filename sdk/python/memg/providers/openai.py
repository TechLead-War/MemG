from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from ..client import MemGClient
from ..intercept import wrap_openai_client
from ..proxy import wrap_openai_proxy

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
    """Wrap an OpenAI client with MemG memory.

    Args:
        client: An OpenAI client instance.
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
        return wrap_openai_proxy(client, entity, proxy_url)
    elif mode == "client":
        if entity is None:
            raise ValueError("entity is required for client mode")
        mcp = MemGClient(mcp_url)
        return wrap_openai_client(client, mcp, entity, extract)
    elif mode == "native":
        if entity is None:
            raise ValueError("entity is required for native mode")
        return _wrap_openai_native(client, entity, extract, config, embedder, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _wrap_openai_native(
    client: Any,
    entity: str,
    extract: bool,
    config: Any,
    embedder: Any,
    **kwargs: Any,
) -> Any:
    """Patch an OpenAI client for native in-process memory."""
    from .. import MemG, MemGConfig

    cfg = config or MemGConfig.from_kwargs(**kwargs)
    engine = MemG(config=cfg, embedder=embedder)
    engine._ensure_native()

    original_create = client.chat.completions.create

    def _patched_create(*args: Any, **call_kwargs: Any) -> Any:
        messages = list(call_kwargs.get("messages") or (args[0] if args else []))
        stream = call_kwargs.get("stream", False)

        from .. import _last_user_content, _inject_context_openai
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

        query = _last_user_content(messages)
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

        augmented = _inject_context_openai(messages, context_str) if context_str else messages

        call_kwargs["messages"] = augmented
        if args:
            args = args[1:]

        if stream:
            return _openai_native_stream(
                original_create, engine, entity, entity_uuid,
                session, extract, cfg, messages, args, call_kwargs, kwargs,
            )

        response = original_create(*args, **call_kwargs)

        content = ""
        try:
            content = response.choices[0].message.content or ""
        except (IndexError, AttributeError):
            pass

        user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        save_exchange(engine._store, session.uuid, entity_uuid, user_msgs, content)

        if extract and content:
            def _bg():
                try:
                    llm_call = engine._make_extraction_llm_call(
                        cfg.llm_provider, cfg.llm_model, kwargs
                    )
                    run_extraction(
                        engine._store, entity_uuid,
                        [m for m in messages if isinstance(m, dict) and m.get("role") == "user"],
                        llm_call, engine._embedder,
                    )
                except Exception as e:
                    logger.warning("memg: extraction failed: %s", e)
            t = threading.Thread(target=_bg, daemon=True)
            t.start()

        return response

    client.chat.completions.create = _patched_create
    client._memg_engine = engine
    return client


def _openai_native_stream(
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
    """Wrap OpenAI streaming with native memory extraction."""
    stream = original_create(*args, **kwargs)

    class _StreamWrapper:
        def __init__(self, inner):
            self._inner = inner
            self._accumulated = []

        def __iter__(self):
            return self

        def __next__(self):
            try:
                chunk = next(self._inner)
            except StopIteration:
                self._finalize()
                raise

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
                m for m in original_messages
                if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            ]
            save_exchange(engine._store, session.uuid, entity_uuid, user_msgs, content)

            if extract and content:
                def _bg():
                    try:
                        llm_call = engine._make_extraction_llm_call(
                            cfg.llm_provider, cfg.llm_model, engine_kwargs
                        )
                        run_extraction(
                            engine._store, entity_uuid,
                            [m for m in original_messages if isinstance(m, dict) and m.get("role") == "user"],
                            llm_call, engine._embedder,
                        )
                    except Exception as e:
                        logger.warning("memg: extraction failed: %s", e)
                t = threading.Thread(target=_bg, daemon=True)
                t.start()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _StreamWrapper(stream)
