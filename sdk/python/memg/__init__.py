"""MemG Python SDK -- memory layer for LLM applications.

Supports three modes:
- native: full in-process engine (SQLite + local embeddings + hybrid search)
- proxy: route through Go proxy server
- client: use Go MCP server via JSON-RPC
"""

from __future__ import annotations

import logging
import math
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from .client import MemGClient, MemGError
from .config import MemGConfig
from .types import (
    AddResult,
    Attribute,
    CanonicalSlot,
    ConsciousFact,
    Fact,
    FactFilter,
    Memory,
    MemoryInput,
    Process,
    RecalledFact,
    SearchResult,
)

from .store import Store, SQLiteStore
from .postgres_store import PostgresStore
from .mysql_store import MySQLStore
from .embedder import (
    Embedder,
    GeminiEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
    auto_detect_embedder,
)
from .search import HybridSearchEngine
from .context import build_context, adaptive_window_size
from .recall import recall_facts, recall_summaries
from .extract import run_extraction
from .artifact_detect import DetectedArtifact, detect_artifacts
from .artifact import store_artifacts, recall_artifacts
from .turn_summary import maintain_turn_summaries
from .entity_mentions import extract_entity_mentions
from .consolidator import consolidate_entity
from .conscious import load_conscious_context
from .decay import prune_expired_and_stale
from .reembed import backfill_missing_embeddings, re_embed_facts
from .summary import generate_and_store_summary
from .conversation import (
    normalize_conversation_messages,
    diff_incoming_messages,
    merge_history,
)
from .query import QueryTransform, QueryTransformer
from .recall_context import recall_and_build_context, RecallConfig

__all__ = [
    # Core
    "MemG",
    "MemGClient",
    "MemGError",
    "MemGConfig",
    # Stores
    "Store",
    "SQLiteStore",
    "PostgresStore",
    "MySQLStore",
    # Embedders
    "Embedder",
    "GeminiEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "create_embedder",
    "auto_detect_embedder",
    # Search & Recall
    "HybridSearchEngine",
    "build_context",
    "recall_facts",
    "recall_summaries",
    "run_extraction",
    # Artifact & Turn Summary
    "DetectedArtifact",
    "detect_artifacts",
    "store_artifacts",
    "recall_artifacts",
    "maintain_turn_summaries",
    "extract_entity_mentions",
    "adaptive_window_size",
    # Types
    "Memory",
    "MemoryInput",
    "AddResult",
    "SearchResult",
    "Fact",
    "FactFilter",
    "RecalledFact",
    "ConsciousFact",
    "CanonicalSlot",
    "Process",
    "Attribute",
    # Consolidation & Conscious
    "consolidate_entity",
    "load_conscious_context",
    # Decay & Reembed
    "prune_expired_and_stale",
    "backfill_missing_embeddings",
    "re_embed_facts",
    # Summary & Conversation
    "generate_and_store_summary",
    "normalize_conversation_messages",
    "diff_incoming_messages",
    "merge_history",
    # Query
    "QueryTransform",
    "QueryTransformer",
]

logger = logging.getLogger("memg")
_EMBEDDER_PROBE_TIMEOUT_S = 15.0


def _probe_embedder(embedder: Embedder) -> None:
    """Verify that the embedder can produce one valid vector."""
    outcome: Dict[str, Any] = {}
    failure: Dict[str, BaseException] = {}
    done = threading.Event()

    def _run_probe() -> None:
        try:
            outcome["vectors"] = embedder.embed(["memg-healthcheck"])
        except BaseException as exc:  # preserve provider exceptions for chaining
            failure["error"] = exc
        finally:
            done.set()

    threading.Thread(target=_run_probe, daemon=True).start()
    if not done.wait(_EMBEDDER_PROBE_TIMEOUT_S):
        raise RuntimeError("memg: embedder health check timed out")
    if "error" in failure:
        raise RuntimeError("memg: embedder health check failed") from failure["error"]
    vectors = outcome.get("vectors")

    if not isinstance(vectors, list) or len(vectors) != 1:
        raise RuntimeError(
            f"memg: embedder health check returned {len(vectors) if isinstance(vectors, list) else 0} vectors"
        )

    vector = vectors[0]
    if not isinstance(vector, list) or not vector:
        raise RuntimeError("memg: embedder health check returned an empty vector")

    expected_dim = embedder.dimension()
    if expected_dim and len(vector) != expected_dim:
        raise RuntimeError(
            f"memg: embedder health check dimension mismatch: got {len(vector)}, want {expected_dim}"
        )

    for idx, value in enumerate(vector):
        if not math.isfinite(value):
            raise RuntimeError(
                f"memg: embedder health check returned invalid value at index {idx}"
            )


def _create_store(config: MemGConfig) -> Store:
    """Create a store from config. Supports sqlite, postgres, mysql."""
    provider = config.store_provider.lower()
    if provider == "postgres" or provider == "postgresql":
        from .postgres_store import PostgresStore
        if not config.store_url:
            raise ValueError("store_url required for postgres (e.g. 'postgresql://user:pass@host/db')")
        return PostgresStore(config.store_url)
    elif provider == "mysql":
        from .mysql_store import MySQLStore
        if not config.store_url:
            raise ValueError("store_url required for mysql (e.g. 'mysql://user:pass@host/db')")
        return MySQLStore(config.store_url)
    else:
        return SQLiteStore(config.db_path)


def _detect_provider(client: Any) -> Optional[str]:
    """Detect the LLM provider from a client instance."""
    module = type(client).__module__
    if "openai" in module:
        return "openai"
    elif "anthropic" in module:
        return "anthropic"
    elif "google" in module or "generativeai" in module:
        return "gemini"
    # Fallback: check for Gemini SDK methods.
    if hasattr(client, "generate_content"):
        return "gemini"
    return None


class MemG:
    """Main entry point for the MemG Python SDK.

    Supports native in-process engine, proxy mode, and MCP client mode.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        config: Optional[MemGConfig] = None,
        embedder: Any = None,
        store: Any = None,
        mcp_url: str = "http://localhost:8686",
        proxy_url: str = "http://localhost:8787/v1",
        **kwargs: Any,
    ) -> None:
        self._config = config or MemGConfig.from_kwargs(**kwargs)
        if db_path:
            self._config.db_path = db_path

        self._mcp_url = mcp_url
        self._proxy_url = proxy_url
        self._mcp_client: Optional[MemGClient] = None

        self._store = store
        self._embedder = embedder
        self._engine = None
        self._initialized = False
        self._init_lock = threading.Lock()
        self._conscious_cache: dict = {}
        self._last_prune_at: float = 0

    def _ensure_native(self) -> None:
        """Lazy-initialize the native engine components."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return

            from .search import HybridSearchEngine
            from .embedder import auto_detect_embedder

            if self._store is None:
                self._store = _create_store(self._config)
            self._engine = HybridSearchEngine()

            if self._embedder is None:
                self._embedder = auto_detect_embedder(
                    api_key=self._config.openai_api_key,
                    model=self._config.embed_model,
                    dimension=self._config.embed_dimension,
                    provider=self._config.embed_provider,
                )
            if self._embedder is None:
                raise RuntimeError(
                    "memg: native mode requires a working embedder. "
                    "Install sentence-transformers or configure a remote embedding provider."
                )
            _probe_embedder(self._embedder)
            self._config.embed_dimension = self._embedder.dimension()

            self._initialized = True

    def _get_mcp_client(self) -> MemGClient:
        if self._mcp_client is None:
            self._mcp_client = MemGClient(self._mcp_url)
        return self._mcp_client

    @staticmethod
    def wrap(
        client: Any,
        entity: Optional[str] = None,
        mode: str = "native",
        proxy_url: str = "http://localhost:8787/v1",
        mcp_url: str = "http://localhost:8686",
        extract: bool = True,
        config: Optional[MemGConfig] = None,
        embedder: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Wrap an LLM client with MemG memory.

        Auto-detects the provider (OpenAI or Anthropic) from the client type.

        Args:
            client: An OpenAI or Anthropic client instance.
            entity: Entity identifier for memory scoping.
            mode: "native" for in-process engine (default),
                  "proxy" to route through MemG proxy,
                  "client" for SDK-side MCP interception.
            proxy_url: MemG proxy URL (proxy mode only).
            mcp_url: MemG MCP server URL (client mode only).
            extract: Whether to extract memories from responses.
            config: MemGConfig for native mode.
            embedder: Custom Embedder instance for native mode.

        Returns:
            A wrapped client instance.
        """
        provider = _detect_provider(client)
        if provider == "openai":
            from .providers.openai import wrap
            return wrap(client, entity, mode, proxy_url, mcp_url, extract, config, embedder, **kwargs)
        elif provider == "anthropic":
            from .providers.anthropic import wrap
            return wrap(client, entity, mode, proxy_url, mcp_url, extract, config, embedder, **kwargs)
        elif provider == "gemini":
            from .providers.gemini import wrap
            return wrap(client, entity, mode, proxy_url, mcp_url, extract, config, embedder, **kwargs)
        else:
            raise ValueError(
                f"Unsupported client type: {type(client).__name__}. "
                "Supported: OpenAI, Anthropic, Gemini"
            )

    # ---- Native memory operations ----

    def _resolve_entity(self, entity_id: str) -> str:
        """Resolve an external entity ID to a UUID, creating if needed."""
        self._ensure_native()
        return self._store.upsert_entity(entity_id)

    def _backfill_missing_embeddings(self, entity_uuid: str) -> None:
        """Backfill embeddings for facts stored without them (e.g., embedder was down)."""
        try:
            unembedded = self._store.list_unembedded_facts(entity_uuid, 50)
            if not unembedded:
                return
            contents = [f.content for f in unembedded]
            embeddings = self._embedder.embed(contents)
            model = self._embedder.model_name()
            for i, f in enumerate(unembedded):
                if i < len(embeddings):
                    try:
                        self._store.update_fact_embedding(f.uuid, embeddings[i], model)
                    except Exception:
                        pass
        except Exception:
            pass

    def _summarize_closed_conversation(self, entity_uuid: str, current_session_uuid: str) -> None:
        """Summarize the most recent unsummarized conversation when a new session starts."""
        conv = self._store.find_unsummarized_conversation(entity_uuid, current_session_uuid)
        if not conv or conv.summary:
            return
        self._generate_and_store_summary(conv.uuid)

    def _generate_and_store_summary(self, conversation_uuid: str) -> None:
        """Generate a summary for a conversation and store it with its embedding."""
        messages = self._store.read_messages(conversation_uuid)
        if not messages:
            return

        transcript = "\n".join(f"{m.role}: {m.content}" for m in messages)

        summary_prompt = (
            "Summarize this conversation. Focus on:\n"
            "- What was discussed\n"
            "- What decisions were made\n"
            "- What is still pending or unresolved\n"
            "- Any new information learned about the user\n\n"
            "Be concise — 2-5 sentences. Only include what is meaningful and worth remembering.\n"
            "If the conversation contains no meaningful content worth remembering "
            "(e.g. just greetings or trivial exchanges), respond with exactly: NONE"
        )

        llm_call = self._make_extraction_llm_call(self._config.llm_provider, self._config.llm_model, {})
        summary = llm_call(summary_prompt, transcript)
        summary = summary.strip()
        if not summary or summary.upper() == "NONE":
            return

        embedding = None
        if self._embedder:
            try:
                vecs = self._embedder.embed([summary])
                if vecs:
                    embedding = vecs[0]
            except Exception:
                pass

        model_name = ""
        if self._embedder and hasattr(self._embedder, "model_name"):
            try:
                model_name = self._embedder.model_name()
            except Exception:
                model_name = ""
        self._store.update_conversation_summary(conversation_uuid, summary, embedding, embedding_model=model_name)

    def add(
        self,
        entity_id: str,
        content_or_memories: Union[str, List[str], List[MemoryInput]],
        **kwargs: Any,
    ) -> AddResult:
        """Add memories for an entity.

        Accepts flexible input:
            - A single string
            - A list of strings
            - A list of MemoryInput
        """
        self._ensure_native()

        memories: List[MemoryInput]
        if isinstance(content_or_memories, str):
            memories = [MemoryInput(
                content=content_or_memories,
                type=kwargs.get("type", "identity"),
                significance=kwargs.get("significance", "medium"),
                tag=kwargs.get("tag"),
            )]
        elif isinstance(content_or_memories, list):
            if not content_or_memories:
                return AddResult(inserted=0, reinforced=0)
            if isinstance(content_or_memories[0], str):
                memories = [
                    MemoryInput(
                        content=c,
                        type=kwargs.get("type", "identity"),
                        significance=kwargs.get("significance", "medium"),
                        tag=kwargs.get("tag"),
                    )
                    for c in content_or_memories
                ]
            elif isinstance(content_or_memories[0], MemoryInput):
                memories = content_or_memories
            else:
                raise TypeError(
                    f"Unsupported list element type: {type(content_or_memories[0]).__name__}"
                )
        else:
            raise TypeError(
                f"Unsupported type for content_or_memories: {type(content_or_memories).__name__}"
            )

        from .extract import content_key, ttl_for_significance, persist_facts, SIGNIFICANCE_HIGH

        entity_uuid = self._resolve_entity(entity_id)
        sig_map = {"low": 1, "medium": 5, "high": 10}

        facts = []
        contents = []
        for m in memories:
            sig = sig_map.get(m.significance, 5) if isinstance(m.significance, str) else int(m.significance)
            f = Fact(
                uuid="",
                content=m.content,
                fact_type=m.type or "identity",
                temporal_status="current",
                significance=sig,
                tag=(m.tag or "").lower().strip(),
                content_key=content_key(m.content),
                expires_at=ttl_for_significance(sig),
                confidence=1.0,
                source_role="user",
                embedding_model=self._embedder.model_name() if self._embedder else "",
            )
            facts.append(f)
            contents.append(m.content)

        if self._embedder and contents:
            try:
                embeddings = self._embedder.embed(contents)
                for i, emb in enumerate(embeddings):
                    if i < len(facts):
                        facts[i].embedding = emb
            except Exception as e:
                logger.warning("memg: embedding failed during add: %s", e)

        inserted, reinforced = persist_facts(self._store, entity_uuid, facts)
        return AddResult(inserted=inserted, reinforced=reinforced)

    def search(
        self, entity_id: str, query: str, limit: int = 10
    ) -> SearchResult:
        """Search memories for an entity using semantic hybrid search."""
        self._ensure_native()

        entity_uuid = self._resolve_entity(entity_id)

        if not self._embedder:
            raise RuntimeError("memg: native mode requires a healthy embedder for memory recall")

        from .recall import recall_facts

        query_vecs = self._embedder.embed([query])
        if not query_vecs:
            raise RuntimeError("memg: embed recall query returned no vectors")

        results = recall_facts(
            self._engine, self._store,
            query_vecs[0], query,
            entity_uuid, limit=limit,
            threshold=self._config.recall_threshold,
            max_candidates=self._config.max_recall_candidates,
        )

        if self._embedder:
            self._backfill_missing_embeddings(entity_uuid)

        if results:
            fact_ids = [r.id for r in results]
            try:
                self._store.update_recall_usage(fact_ids)
            except Exception:
                pass

        memories = [
            Memory(
                id=r.id,
                content=r.content,
                type="identity",
                temporal_status=r.temporal_status,
                significance=_significance_label(r.significance),
                score=r.score,
                created_at=r.created_at,
            )
            for r in results
        ]
        return SearchResult(memories=memories, count=len(memories))

    def list(
        self,
        entity_id: str,
        limit: int = 50,
        type: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> SearchResult:
        """List memories for an entity, optionally filtered."""
        self._ensure_native()
        entity_uuid = self._resolve_entity(entity_id)

        filt = FactFilter()
        if type:
            filt.types = [type]
        if tag:
            filt.tags = [tag]

        facts = self._store.list_facts_filtered(entity_uuid, filt, limit)
        memories = [
            Memory(
                id=f.uuid,
                content=f.content,
                type=f.fact_type,
                temporal_status=f.temporal_status,
                significance=_significance_label(f.significance),
                created_at=f.created_at,
                tag=f.tag or None,
                reinforced_count=f.reinforced_count,
            )
            for f in facts
        ]
        return SearchResult(memories=memories, count=len(memories))

    def delete(self, entity_id: str, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        self._ensure_native()
        entity_uuid = self._resolve_entity(entity_id)
        self._store.delete_fact(entity_uuid, memory_id)
        return True

    def delete_all(self, entity_id: str) -> int:
        """Delete all memories for an entity."""
        self._ensure_native()
        entity_uuid = self._resolve_entity(entity_id)
        return self._store.delete_entity_facts(entity_uuid)

    def chat(
        self,
        messages: List[Dict[str, str]],
        entity: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Memory-augmented chat (native mode).

        Recalls, injects context, forwards to LLM, extracts facts.
        Requires openai or anthropic package for the LLM call.

        Returns a dict with 'content', 'role', and 'memory_context'.
        """
        self._ensure_native()

        if not entity:
            raise ValueError("entity is required for chat()")

        entity_uuid = self._resolve_entity(entity)
        process_id = kwargs.get("process_id", "default")

        from .session import ensure_session, get_or_create_conversation, save_exchange, load_recent_history
        from .recall import recall_facts, recall_summaries
        from .context import build_context
        from .extract import run_extraction

        session, is_new = ensure_session(
            self._store, entity_uuid, process_id, self._config.session_timeout
        )

        if is_new:
            import threading
            threading.Thread(
                target=self._summarize_closed_conversation,
                args=(entity_uuid, session.uuid),
                daemon=True,
            ).start()

        history = load_recent_history(
            self._store, session.uuid, self._config.working_memory_turns
        )

        conscious = []
        if self._config.conscious_mode:
            conscious = self._load_conscious_facts_cached(entity_uuid)

        self._prune_if_due(entity_uuid)

        query = _last_user_content(messages)
        recalled = []
        summaries_recalled = []

        if query:
            if not self._embedder:
                raise RuntimeError("memg: native mode requires a healthy embedder for memory recall")
            query_vecs = self._embedder.embed([query])
            if not query_vecs:
                raise RuntimeError("memg: embed recall query returned no vectors")
            query_vec = query_vecs[0]
            recalled = recall_facts(
                self._engine, self._store,
                query_vec, query, entity_uuid,
                limit=self._config.recall_limit,
                threshold=self._config.recall_threshold,
                max_candidates=self._config.max_recall_candidates,
            )
            summaries_recalled = recall_summaries(
                self._engine, self._store,
                query_vec, query, entity_uuid,
                limit=5, threshold=self._config.recall_threshold,
            )
            if recalled:
                fact_ids = [r.id for r in recalled]
                try:
                    self._store.update_recall_usage(fact_ids)
                except Exception:
                    pass

        context_str = build_context(
            conscious_facts=conscious,
            recalled_facts=recalled,
            summaries=summaries_recalled,
            total_token_budget=self._config.memory_token_budget,
            summary_token_budget=self._config.summary_token_budget,
        )

        augmented_messages = _inject_context_openai(messages, context_str) if context_str else list(messages)

        llm_provider = kwargs.get("llm_provider", self._config.llm_provider)
        llm_model = kwargs.get("model", self._config.llm_model)
        response_content = self._call_llm(augmented_messages, llm_provider, llm_model, kwargs)

        user_msgs = [m for m in messages if m.get("role") in ("user", "assistant")]
        save_exchange(
            self._store, session.uuid, entity_uuid,
            user_msgs, response_content,
        )

        if self._config.extract and response_content:
            def _bg_extract():
                try:
                    llm_call = self._make_extraction_llm_call(llm_provider, llm_model, kwargs)
                    run_extraction(
                        self._store, entity_uuid, user_msgs,
                        llm_call, self._embedder,
                    )
                except Exception as e:
                    logger.warning("memg: background extraction failed: %s", e)

            t = threading.Thread(target=_bg_extract, daemon=True)
            t.start()

        return {
            "content": response_content,
            "role": "assistant",
            "memory_context": context_str,
        }

    def _load_conscious_facts_cached(self, entity_uuid: str) -> List[ConsciousFact]:
        import time as _time
        now = _time.monotonic()
        cached = self._conscious_cache.get(entity_uuid)
        if cached and cached["expires_at"] > now:
            return cached["facts"]
        facts = self._load_conscious_facts(entity_uuid)
        self._conscious_cache[entity_uuid] = {"facts": facts, "expires_at": now + 30}
        return facts

    def _prune_if_due(self, entity_uuid: str) -> None:
        import time as _time
        now = _time.monotonic()
        if now - self._last_prune_at < 300:  # 5 min interval
            return
        self._last_prune_at = now
        try:
            self._store.prune_expired_facts(entity_uuid)
        except Exception:
            pass

    def _load_conscious_facts(self, entity_uuid: str) -> List[ConsciousFact]:
        """Load top facts by significance for conscious mode."""
        filt = FactFilter(statuses=["current"], exclude_expired=True)
        fetch_limit = max(50, self._config.conscious_limit * 5)
        facts = self._store.list_facts_filtered(entity_uuid, filt, fetch_limit)
        if not facts:
            return []

        now = datetime.now(timezone.utc)
        scored = []
        for f in facts:
            base = float(f.significance)
            if f.fact_type == "identity" and f.significance < 10:
                last_confirmed = f.created_at
                if f.reinforced_at and f.reinforced_at > last_confirmed:
                    last_confirmed = f.reinforced_at
                if f.last_recalled_at and f.last_recalled_at > last_confirmed:
                    last_confirmed = f.last_recalled_at
                if last_confirmed:
                    days_since = (now - last_confirmed).total_seconds() / 86400
                    if days_since > 30:
                        staleness = (days_since - 30) / 90
                        if staleness > 0.5:
                            staleness = 0.5
                        base *= (1 - staleness)
            scored.append((f, base))

        scored.sort(key=lambda x: (-x[1], x[0].uuid))

        limit = self._config.conscious_limit
        if len(scored) > limit:
            scored = scored[:limit]

        return [
            ConsciousFact(
                id=f.uuid,
                content=f.content,
                significance=f.significance,
                tag=f.tag,
            )
            for f, _ in scored
        ]

    def _call_llm(self, messages: list, provider: str, model: str, kwargs: dict) -> str:
        """Call the LLM provider and return response content."""
        if provider == "openai":
            return self._call_openai(messages, model, kwargs)
        elif provider == "anthropic":
            return self._call_anthropic(messages, model, kwargs)
        elif provider == "gemini":
            return self._call_gemini(messages, model, kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def _call_openai(self, messages: list, model: str, kwargs: dict) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        api_key = kwargs.get("api_key") or self._config.openai_api_key
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 4096),
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, messages: list, model: str, kwargs: dict) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        client = anthropic.Anthropic(api_key=api_key)

        system = ""
        chat_msgs = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                chat_msgs.append(m)

        create_kwargs = {
            "model": model,
            "messages": chat_msgs,
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        if system:
            create_kwargs["system"] = system

        resp = client.messages.create(**create_kwargs)
        for block in resp.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def _call_gemini(self, messages: list, model: str, kwargs: dict) -> str:
        import httpx

        api_key = kwargs.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY env var or pass api_key.")

        system_text = ""
        contents = []
        for m in messages:
            if m.get("role") == "system":
                system_text = m.get("content", "")
            else:
                role = "model" if m.get("role") == "assistant" else m.get("role", "user")
                contents.append({"role": role, "parts": [{"text": m.get("content", "")}]})

        body: Dict[str, Any] = {"contents": contents, "generationConfig": {"temperature": 0.1}}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        resp = httpx.post(url, json=body, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    def _make_extraction_llm_call(self, provider: str, model: str, kwargs: dict):
        """Create a callable for extraction LLM calls."""
        def _call(system_prompt: str, user_content: str) -> str:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            return self._call_llm(messages, provider, model, kwargs)
        return _call

    def close(self) -> None:
        """Close the underlying store and client connections."""
        if self._store:
            self._store.close()
        if self._mcp_client:
            self._mcp_client.close()


def _significance_label(sig: int) -> str:
    if sig >= 10:
        return "high"
    elif sig >= 5:
        return "medium"
    return "low"


def _last_user_content(messages: list) -> Optional[str]:
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


def _inject_context_openai(messages: list, context_text: str) -> list:
    """Inject memory context into OpenAI-style messages."""
    messages = [m.copy() if isinstance(m, dict) else m for m in messages]
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "system":
            existing = msg.get("content", "")
            messages[i] = {**msg, "content": existing + "\n\n" + context_text}
            return messages
    messages.insert(0, {"role": "system", "content": context_text})
    return messages
