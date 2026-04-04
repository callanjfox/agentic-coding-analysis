# Trace Format Specification

Traces are compact JSON files that capture the cache block structure of real Claude Code conversations. They contain everything needed to simulate cache behavior without storing actual message content.

## Top-Level Schema

```json
{
  "id": "8712b46f-04e",
  "models": ["claude-sonnet-4-20250514"],
  "block_size": 64,
  "tool_tokens": 11880,
  "system_tokens": 3427,
  "requests": [...],
  "_analysis": {
    "indexed": 1341,
    "total": 2363,
    "unique_blocks": 9879
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | First 12 chars of conversation UUID (+ `_s1`, `_s2` if split) |
| `models` | string[] | All model IDs used in conversation |
| `block_size` | int | Token block size (64 recommended) |
| `tool_tokens` | int | Token count for tools section (shared prefix) |
| `system_tokens` | int | Token count for system prompt (shared prefix) |
| `requests` | array | Ordered list of request records |
| `_analysis` | object | Build metadata (indexed count, total requests, unique blocks) |

### Shared Prefix Fields

`tool_tokens` and `system_tokens` represent the shared prefix (~15K tokens) that stays warm in the API's global cache across all Claude Code sessions. For replay, optionally pre-populate cache with the first N blocks:

```python
warm_prefix_blocks = (trace['tool_tokens'] + trace['system_tokens']) // trace['block_size']
```

## Request Schema

```json
{
  "t": 0.0,
  "type": "s",
  "model": "claude-sonnet-4-20250514",
  "in": 19105,
  "out": 297,
  "hash_ids": [1, 2, 3, "...", 298],
  "input_types": ["text"],
  "output_types": ["thinking", "text", "tool_use"],
  "stop": "tool_use"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `t` | float | Seconds from conversation start |
| `type` | string | `"s"` = streaming, `"n"` = non-streaming, `"subagent"` = nested |
| `model` | string | Model ID for this request |
| `in` | int | Input token count |
| `out` | int | Output token count |
| `hash_ids` | int[] | Ordered block hash IDs for prefix-based cache matching |
| `input_types` | string[] | What the client added this turn |
| `output_types` | string[] | Content types in Claude's response |
| `stop` | string | Stop reason: `""`, `"tool_use"`, `"end_turn"` |
| `api_time` | float? | Total response time in seconds (from proxy `responseTime`) |
| `ttft` | float? | Time to first token in seconds (from `Server-Timing` header). Only present on streaming requests (`type: "s"`), where it captures server-side latency before the first token. Omitted on non-streaming requests since TTFT equals `api_time` |
| `think_time` | float? | Client delay before this request in seconds (gap since previous response completed) |

### input_types Values

Shows what the **client added** for this specific request (not cumulative):

| Condition | input_types |
|-----------|-------------|
| First request | `["text"]` |
| Within-turn pair (s→n) | `[]` |
| Cross-turn after `tool_use` | `["tool_result"]` |
| Cross-turn after `end_turn` | `["text"]` |
| Major context reset (<50% hash preserved) | Full content types from messages |

### output_types Values

Content types Claude returned:
- `"text"` — Text content
- `"thinking"` — Extended thinking block
- `"tool_use"` — Tool call

## Hash ID Generation Algorithm

### Steps

1. **Extract** content sections in order: tools → system → messages
2. **Normalize** — recursively remove `cache_control` and `signature` fields
3. **Serialize** — `json.dumps(obj, separators=(',', ':'))`
4. **Tokenize** — using tiktoken `gpt-4` encoding
5. **Block** — split into `block_size` token blocks (**full blocks only**, partial discarded)
6. **Chain hash** — each block depends on the previous block's hash

```python
def create_chained_hash(token_ids, prev_hash, seq_num):
    content = f"{prev_hash}:{seq_num}:" + " ".join(map(str, token_ids))
    return hashlib.sha256(content.encode()).hexdigest()[:16]
```

### Hash ID Assignment

Each unique hash string gets a sequential integer ID (starting at 1). Currently, IDs are assigned **per-conversation** — the same content in different conversations gets different numeric IDs.

### Why Full Blocks Only

Partial blocks (< block_size tokens) at the end are discarded because the last partial chunk changes every turn as content grows. This would cause false cache misses. Full blocks provide stable, reproducible hashes.

### Why Chained Hashing

Each block's hash includes the previous block's hash, which encodes **position**. This means identical content at different positions produces different hashes — matching real cache behavior where prefix position matters.

## Request Pairing Pattern

Every conversation turn produces **two requests** with identical `hash_ids`:

1. **Streaming** (first): `max_tokens=32000, stream=true` — for UI responsiveness
2. **Non-streaming** (5-8s later): `max_tokens=21333, stream=false` — for tool execution

The non-streaming request gets ~100% cache hit from the streaming request's cache creation.

## Hash Evolution Patterns

From real conversation data:

| Pattern | Frequency | Description |
|---------|-----------|-------------|
| Identical (100% prefix) | 97.8% | Within-turn pairs (s→n) |
| Suffix change (90-99%) | 0.5% | Normal growth + tokenization boundary effects |
| Major reset (<50%) | 1.7% | Context restructure or conversation restart |

### The "Last Chunk Different" Pattern

When `cache_control` moves to a new last block between requests, the previous last block's hash changes because normalization removes `cache_control` from the content being hashed. This causes exactly one block to differ at the boundary — this is expected behavior.

## Conversation Splitting

When generated with `--split-at-gap`, long conversations are split when the time gap between requests exceeds a threshold:

- Trace IDs get `_s1`, `_s2`, etc. suffix (e.g., `f99f3001-351_s1`)
- Each segment has its own `tool_tokens`/`system_tokens` from its first request
- Each segment's `t` values restart from 0.0

## Sub-Agent Format

When generated with `--include-subagents`, traces can contain nested sub-agent conversations:

```json
{
  "t": 82075.0,
  "type": "subagent",
  "agent_id": "96624242",
  "subagent_type": "Explore",
  "duration_ms": 120012,
  "total_tokens": 83681,
  "tool_use_count": 22,
  "status": "completed",
  "models": ["claude-haiku-4-5-20251001"],
  "tool_tokens": 7810,
  "system_tokens": 730,
  "requests": [
    {"t": 0.0, "type": "s", "in": 9969, "out": 247, "hash_ids": [...]},
    {"t": 3.1, "type": "n", "in": 9969, "out": 247, "hash_ids": [...]}
  ]
}
```

### Sub-Agent Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Always `"subagent"` |
| `t` | float | When spawned (seconds from parent conversation start) |
| `agent_id` | string | 7-8 char identifier |
| `subagent_type` | string | "Explore", "general-purpose", "Plan", etc. |
| `duration_ms` | int | Total execution time in milliseconds |
| `total_tokens` | int | Total tokens consumed |
| `tool_use_count` | int | Number of tool calls made |
| `status` | string | "completed", "failed", or "timeout" |
| `models` | string[] | Models used (often different from parent) |
| `tool_tokens` | int | Sub-agent's tool definition tokens |
| `system_tokens` | int | Sub-agent's system prompt tokens |
| `requests` | array | Nested request array (same schema as parent) |

### Aggregate Totals

Traces with sub-agents include a `totals` field:

```json
{
  "totals": {
    "parent_tokens": {"input": 48953966, "output": 127608},
    "subagent_tokens": {"input": 1146338, "output": 10819},
    "combined_tokens": {"input": 50100304, "output": 138427},
    "subagent_count": 1
  }
}
```

### Sub-Agent Cache Behavior

Sub-agents run in **isolated cache contexts**:
- They start fresh — they do NOT inherit the parent's cache
- Only the global prefix (tools + system, ~8-12K tokens) may be warm
- Parent's cache remains warm during sub-agent execution
- Sub-agent's cache is discarded after completion

```python
def simulate_cache(trace):
    parent_cache = set()

    for req in trace['requests']:
        if req['type'] == 'subagent':
            subagent_cache = set()  # Fresh cache
            for subreq in req['requests']:
                hits = count_prefix_match(subreq['hash_ids'], subagent_cache)
                subagent_cache.update(subreq['hash_ids'])
            # subagent_cache discarded here
        else:
            hits = count_prefix_match(req['hash_ids'], parent_cache)
            parent_cache.update(req['hash_ids'])
```

## Backward Compatibility

Traces without sub-agents have no `type: "subagent"` requests and no `totals` field. To check:

```python
has_subagents = any(r.get('type') == 'subagent' for r in trace['requests'])
```
