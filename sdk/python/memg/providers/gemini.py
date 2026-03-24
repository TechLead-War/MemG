"""Gemini provider wrapping for MemG.

Supports three modes:
- native: full in-process engine
- client: intercepts calls locally via MCP
- proxy: not supported for Gemini (falls back to native)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

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
    """Wrap a Gemini GenerativeModel with MemG memory."""
    if mode == "client":
        from ..client import MemGClient
        from ..intercept import wrap_gemini_client
        mcp_client = MemGClient(mcp_url)
        return wrap_gemini_client(client, mcp_client, entity or "default", extract)

    if mode == "proxy":
        logger.warning("memg: proxy mode not supported for Gemini, using native mode")

    return _wrap_native(client, entity, extract, config, embedder, **kwargs)


def _wrap_native(
    client: Any,
    entity: Optional[str],
    extract: bool,
    config: Any,
    embedder: Any,
    **kwargs: Any,
) -> Any:
    """Wrap with the native in-process engine."""
    from .. import MemG as MemGEngine
    from ..config import MemGConfig

    cfg = config or MemGConfig.from_kwargs(**kwargs)
    memg = MemGEngine(config=cfg, embedder=embedder)

    original_generate = client.generate_content

    def _patched_generate(*args: Any, **kw: Any) -> Any:
        memg._ensure_native()
        entity_id = entity or "default"

        # Normalize input
        prompt = args[0] if args else kw.get("contents", "")
        if isinstance(prompt, str):
            contents = [{"role": "user", "parts": [{"text": prompt}]}]
        elif isinstance(prompt, list):
            contents = prompt
        else:
            contents = prompt

        # Step 1: Build memory context
        user_text = _last_user_text(contents)
        context_str = ""
        if user_text:
            context_str = memg.search(entity_id, user_text, limit=10)
            if context_str and context_str.memories:
                lines = [f"- {m.content}" for m in context_str.memories]
                context_str = (
                    "[Memory Context]\n"
                    "The following are relevant memories about this user:\n"
                    + "\n".join(lines)
                    + "\n[End Memory Context]"
                )
            else:
                context_str = ""

        # Inject context into systemInstruction if available
        if context_str:
            existing_sys = kw.get("system_instruction", getattr(client, "_system_instruction", None))
            if existing_sys:
                existing_text = str(existing_sys)
                kw["system_instruction"] = existing_text + "\n\n" + context_str
            else:
                kw["system_instruction"] = context_str

        # Step 2: Call original
        if args:
            response = original_generate(*args, **kw)
        else:
            response = original_generate(**kw)

        # Step 3: Background extraction
        if extract and entity:
            try:
                resp_text = response.text if hasattr(response, "text") else ""
                if resp_text:
                    def _bg():
                        try:
                            messages = []
                            if user_text:
                                messages.append({"role": "user", "content": user_text})
                            messages.append({"role": "assistant", "content": resp_text})
                            from ..extract import run_extraction
                            llm_call = memg._make_extraction_llm_call(
                                "gemini",
                                getattr(client, "model_name", "gemini-2.5-flash"),
                                kwargs,
                            )
                            entity_uuid = memg._resolve_entity(entity_id)
                            run_extraction(
                                memg._store, entity_uuid, messages, llm_call, memg._embedder,
                            )
                        except Exception as e:
                            logger.warning("memg: background extraction failed: %s", e)

                    t = threading.Thread(target=_bg, daemon=True)
                    t.start()
            except Exception:
                pass

        return response

    client.generate_content = _patched_generate
    client._memg = memg
    client._memg_close = lambda: memg.close()
    return client


def _last_user_text(contents: Any) -> Optional[str]:
    """Extract last user text from Gemini contents."""
    if not isinstance(contents, list):
        if isinstance(contents, str):
            return contents
        return None
    for c in reversed(contents):
        if isinstance(c, dict) and c.get("role") == "user":
            parts = c.get("parts", [])
            if parts and isinstance(parts[0], dict):
                return parts[0].get("text")
    return None
