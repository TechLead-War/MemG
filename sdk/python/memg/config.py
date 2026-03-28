"""Configuration for MemG native engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class MemGConfig:
    """Configuration for the MemG native engine."""

    store_provider: str = "sqlite"  # sqlite, postgres, mysql
    store_url: str = ""  # connection string for postgres/mysql
    db_path: str = "memg.db"
    embed_provider: str = "sentence-transformers"
    embed_model: str = "Xenova/all-MiniLM-L6-v2"
    embed_dimension: int = 384
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    recall_limit: int = 100
    recall_threshold: float = 0.10
    max_recall_candidates: int = 50
    session_timeout: int = 1800  # 30 minutes in seconds
    working_memory_turns: int = 10
    memory_token_budget: int = 4000
    summary_token_budget: int = 1000
    conscious_mode: bool = True
    conscious_limit: int = 10
    extract: bool = True
    openai_api_key: str = ""
    gemini_api_key: str = ""

    def __post_init__(self) -> None:
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.gemini_api_key:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")

    @staticmethod
    def from_kwargs(**kwargs) -> MemGConfig:
        """Create config from keyword arguments, ignoring unknown keys."""
        valid_fields = {f.name for f in MemGConfig.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return MemGConfig(**filtered)
