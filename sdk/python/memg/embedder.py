"""Embedder interface and implementations for MemG."""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger("memg")


class Embedder(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts into dense vectors."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    def model_name(self) -> str:
        """Return the model identifier."""
        return ""


class SentenceTransformerEmbedder(Embedder):
    """Local embeddings using sentence-transformers. No API keys needed."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install memg-sdk[local]"
            )
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        self._name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._name


class OpenAIEmbedder(Embedder):
    """API embeddings using OpenAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )
        self._model = model
        self._dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": texts,
        }
        if self._dim and self._model.startswith("text-embedding-3"):
            payload["dimensions"] = self._dim

        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()

        results = sorted(data["data"], key=lambda x: x["index"])
        return [r["embedding"] for r in results]

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._model


class GeminiEmbedder(Embedder):
    """API embeddings using Google Gemini's batchEmbedContents endpoint."""

    MAX_BATCH_SIZE = 100

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-004",
        dim: int = 768,
        base_url: str = "https://generativelanguage.googleapis.com",
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key."
            )
        self._model = model
        self._dim = dim
        self._base_url = base_url.rstrip("/")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results: List[List[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[start : start + self.MAX_BATCH_SIZE]
            results.extend(self._embed_batch(batch))
        return results

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        import httpx

        model_ref = f"models/{self._model}"
        url = f"{self._base_url}/v1beta/{model_ref}:batchEmbedContents?key={self._api_key}"

        payload = {
            "requests": [
                {
                    "model": model_ref,
                    "content": {"parts": [{"text": text}]},
                }
                for text in texts
            ]
        }

        resp = httpx.post(url, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()

        embeddings = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Gemini: expected {len(texts)} embeddings, got {len(embeddings)}"
            )
        return [e["values"] for e in embeddings]

    def dimension(self) -> int:
        return self._dim

    def model_name(self) -> str:
        return self._model


def create_embedder(
    provider: str = "sentence-transformers",
    model: Optional[str] = None,
    dimension: Optional[int] = None,
    api_key: Optional[str] = None,
) -> Embedder:
    """Create an embedder by provider name.

    Tries sentence-transformers first for local, falls back to OpenAI.
    """
    if provider == "sentence-transformers":
        model_name = model or "all-MiniLM-L6-v2"
        return SentenceTransformerEmbedder(model_name)
    elif provider == "openai":
        return OpenAIEmbedder(
            api_key=api_key,
            model=model or "text-embedding-3-small",
            dim=dimension or 1536,
        )
    elif provider == "gemini":
        return GeminiEmbedder(
            api_key=api_key,
            model=model or "text-embedding-004",
            dim=dimension or 768,
        )
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")


def auto_detect_embedder(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    dimension: Optional[int] = None,
    provider: Optional[str] = None,
) -> Optional[Embedder]:
    """Try to create an embedder, preferring local over API.

    Returns None if no embedder can be initialized.
    """
    if provider:
        try:
            return create_embedder(
                provider=provider,
                model=model,
                dimension=dimension,
                api_key=api_key,
            )
        except (ImportError, ValueError) as e:
            logger.warning("memg: failed to create %s embedder: %s", provider, e)
            return None

    try:
        return SentenceTransformerEmbedder(model or "all-MiniLM-L6-v2")
    except ImportError:
        pass

    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if key:
        try:
            return OpenAIEmbedder(
                api_key=key,
                model=model or "text-embedding-3-small",
                dim=dimension or 1536,
            )
        except ValueError:
            pass

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        try:
            return GeminiEmbedder(
                api_key=gemini_key,
                model=model or "text-embedding-004",
                dim=dimension or 768,
            )
        except ValueError:
            pass

    return None
