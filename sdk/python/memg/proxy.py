from __future__ import annotations

from typing import Optional, Any


def wrap_openai_proxy(
    client: Any,
    entity: Optional[str] = None,
    proxy_url: str = "http://localhost:8787/v1",
) -> Any:
    """Wrap an OpenAI client to route through the MemG proxy.

    Returns a new client instance with the base URL and headers configured.
    The original client is not mutated.
    """
    headers: dict = {}
    if entity:
        headers["X-MemG-Entity"] = entity
    return client.with_options(base_url=proxy_url, default_headers=headers)


def wrap_anthropic_proxy(
    client: Any,
    entity: Optional[str] = None,
    proxy_url: str = "http://localhost:8787/v1",
) -> Any:
    """Wrap an Anthropic client to route through the MemG proxy.

    Returns a new client instance with the base URL and headers configured.
    The original client is not mutated.
    """
    headers: dict = {}
    if entity:
        headers["X-MemG-Entity"] = entity
    return client.with_options(base_url=proxy_url, default_headers=headers)
