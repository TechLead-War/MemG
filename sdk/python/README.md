# MemG Python SDK

Python SDK for [MemG](https://github.com/primetrace/memg) -- a pluggable memory layer for LLM applications.

## Installation

```bash
pip install memg

# With provider extras
pip install memg[openai]
pip install memg[anthropic]
pip install memg[gemini]
pip install memg[all]
```

## Quick Start

### Proxy Mode (simplest)

Redirect your LLM client through the MemG proxy. No extra code needed beyond wrapping:

```python
from openai import OpenAI
from memg import MemG

client = OpenAI()
client = MemG.wrap(client, entity="user-123", mode="proxy")

# Use as normal -- MemG injects and extracts memories transparently
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "I love hiking in the mountains"}],
)
```

### Client Mode (no proxy needed)

The SDK intercepts calls, queries the MCP server for relevant memories, injects them, and extracts new knowledge:

```python
from openai import OpenAI
from memg import MemG

client = OpenAI()
client = MemG.wrap(client, entity="user-123", mode="client")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Remember I prefer dark roast coffee"}],
)
```

### Direct Memory Operations

```python
from memg import MemG

m = MemG()

# Add memories
m.add("user-123", "likes coffee")
m.add("user-123", ["works at Acme", "prefers dark mode"])

# Search
results = m.search("user-123", "coffee preferences")
for mem in results.memories:
    print(f"{mem.content} (score={mem.score})")

# List
all_mems = m.list("user-123", type="identity")

# Delete
m.delete("user-123", memory_id="some-uuid")
m.delete_all("user-123")

m.close()
```

### Chat (Session-Aware)

`chat()` manages the full memory loop: sessions, history injection, recall, LLM call, exchange persistence, and extraction.

```python
from memg import MemG

m = MemG(embed_provider="openai", openai_api_key="...")
res = m.chat(
    [{"role": "user", "content": "I just moved to Seattle"}],
    entity_id="user-123"
)
print(res["content"])

# Follow-ups are history-aware — the session tracks prior turns.
res2 = m.chat(
    [{"role": "user", "content": "What city did I mention?"}],
    entity_id="user-123"
)
```

### Gemini

```python
import google.generativeai as genai
from memg import MemG

genai.configure(api_key="...")
model = genai.GenerativeModel("gemini-2.5-flash")
model = MemG.wrap(model, entity="user-123")

response = model.generate_content("What do you remember about me?")
```

### Custom Store (MySQL, Postgres, etc.)

The SDK defaults to SQLite, but you can pass any object that implements the `Store` interface:

```python
from memg import MemG, Store

class MyPostgresStore(Store):
    # Implement all abstract methods...
    pass

m = MemG(store=MyPostgresStore(connection_string))
```

See `memg.store.Store` for the full interface contract (27 methods).

## Modes

| Mode | Requires | How it works |
|------|----------|--------------|
| `native` | Nothing | Full in-process engine with SQLite (default) |
| `proxy` | MemG proxy running | Redirects LLM calls through the proxy via `with_options()` |
| `client` | MemG MCP server running | SDK intercepts calls, queries MCP for memories, injects context |

## Supported Providers

| Provider | `wrap()` | Extraction | Embeddings |
|----------|----------|------------|------------|
| OpenAI | `openai.OpenAI()` | Yes | Yes |
| Anthropic | `anthropic.Anthropic()` | Yes | No |
| Gemini | `genai.GenerativeModel()` | Yes | Yes |

## Configuration

Default URLs:
- MCP server: `http://localhost:8686`
- Proxy: `http://localhost:8787/v1`

Override via constructor or `wrap()`:

```python
m = MemG(mcp_url="http://custom:8686", proxy_url="http://custom:8787/v1")
MemG.wrap(client, mode="proxy", proxy_url="http://custom:8787/v1")
```
