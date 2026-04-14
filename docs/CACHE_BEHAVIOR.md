# Claude Cache Behavior

This document explains how Claude's prompt caching works, how Claude Code interacts with it, and what our analysis has revealed.

## Claude Messages API Structure

### Request Body

```json
{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 32000,
  "stream": true,
  "system": [{"type": "text", "text": "System prompt...", "cache_control": {"type": "ephemeral"}}],
  "tools": [{"name": "Read", "description": "...", "input_schema": {...}}],
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "..."}]},
    {"role": "assistant", "content": [{"type": "thinking", "thinking": "...", "signature": "..."}, ...]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01ABC", "content": "..."}]}
  ]
}
```

Content is tokenized and cached in order: **tools → system → messages**.

### Response Body

```json
{
  "id": "msg_01XYZ...",
  "content": [
    {"type": "thinking", "thinking": "...", "signature": "Eo8ECkYI..."},
    {"type": "text", "text": "..."},
    {"type": "tool_use", "id": "toolu_01ABC", "name": "Read", "input": {"path": "/file"}}
  ],
  "stop_reason": "tool_use",
  "usage": {
    "input_tokens": 9,
    "output_tokens": 500,
    "cache_read_input_tokens": 45000,
    "cache_creation_input_tokens": 5000
  }
}
```

### Cache Metrics Formula

```
total_input = input_tokens + cache_read_input_tokens + cache_creation_input_tokens
```

- `cache_read_input_tokens` — tokens served from cache (fast, cheaper)
- `cache_creation_input_tokens` — tokens added to cache this request
- `input_tokens` — tokens that were neither cached nor created (typically ~9-12 per request)

## How Prefix Caching Works

Claude's cache is **prefix-based**: content is cached from the beginning of the request, and cache hits occur when the prefix of a new request matches a previously cached prefix.

```
Turn 1: [tools][system][user_msg]          → all new, creates cache
Turn 2: [tools][system][user_msg][asst][tool_result] → prefix matches turn 1
         ─── cached ────────────  ─── new ────────
```

If any block in the prefix differs from the cached version, all subsequent blocks also miss. There is no "skip and match later" — it's strictly prefix-based.

## cache_control Mechanics

Claude Code places `cache_control: {"type": "ephemeral"}` on the last content block to hint where caching should extend to. This marker **moves** each request:

```
Turn 1: tools[...] system[..., cache_control] messages[user_text]
Turn 2: tools[...] system[...] messages[user_text, cache_control, asst, tool_result]
```

### The "Last Chunk Different" Pattern

When `cache_control` moves, the block that previously contained it gets a different hash (because we normalize by removing `cache_control` before hashing). This causes exactly **one block** to differ at the boundary — a known and expected artifact.

```
Turn 1 hash_ids: [1, 2, 3, ..., 71, 72]        (72 has cache_control)
Turn 2 hash_ids: [1, 2, 3, ..., 71, 73, 74...]  (72→73 without cache_control)
```

## Thinking Block Signatures

Assistant responses include `signature` fields on thinking blocks (256-20,528 chars, avg 688). These change every request even for identical thinking content. We **normalize them out** before hashing to prevent false cache misses.

## Cross-Conversation Global Cache

The API maintains a **global cache** shared across all Claude Code sessions. This means:

- Tool definitions (~12K tokens) are always warm from other active sessions
- System prompt (~3K tokens) is always warm
- The first request of any new conversation already sees ~100% cache hit

Evidence:
```
First request: 72,346/72,355 cached (99.99%), only 9 tokens uncached
First request: 72,531/72,534 cached (99.99%), only 3 tokens uncached
```

The 3-10 uncached tokens are just the user's specific input text. The entire ~72K prefix is already warm.

### Impact on Simulation

Our simulation only models within-conversation caching. Three factors contribute to the ~1.9 percentage point gap:

| Metric | With Infinite TTL | API Actual |
|--------|-------------------|------------|
| Simulated avg | ~97.7% | ~99.6% |
| Gap | — | ~1.9pp |

**1. Cross-conversation global cache (~1-2pp)** — The global cache keeps the shared prefix warm across sessions, which our per-conversation simulation cannot model.

**2. Tokenizer differences** — We use tiktoken's GPT-4 tokenizer as an approximation. Claude's actual tokenizer produces slightly different token boundaries, causing minor block alignment mismatches.

**3. Full-block rounding** — We only simulate full 64-token blocks, discarding partial remainders (0-63 tokens per request). The API caches at a finer granularity, so our block-level simulation systematically undercounts cached tokens by a small amount.

## TTL Behavior

Cache entries have a time-to-live (TTL). Our analysis found:

- **5-minute TTL** caused ~2.5pp of false cache misses (gaps > 5 min between requests)
- **Infinite TTL** eliminates these false misses, giving ~97.7% simulated rate
- The API's actual TTL appears to be longer than 5 minutes for actively-used sessions

Recommendation: use infinite TTL for measuring maximum achievable cache rate.

## Major Context Resets

Occasionally, the conversation context changes dramatically:

| Type | Hash Preservation | Cause |
|------|-------------------|-------|
| Normal growth | 97-100% | New content appended to prefix |
| Major reset | <50% | Context restructure, conversation restart, or context window overflow |

Major resets are detected when `<50%` of previous hash_ids are preserved. They represent events like conversation truncation or system prompt changes.

## Validated Cache Statistics

From analysis of real Claude Code sessions:

| Metric | Value |
|--------|-------|
| Overall cache hit rate | 85-90% |
| Within-conversation prefix reuse | 93-99% |
| Cross-conversation benefit (system/tools) | ~1.9pp |
| Uncached overhead per request | ~9-12 tokens |
