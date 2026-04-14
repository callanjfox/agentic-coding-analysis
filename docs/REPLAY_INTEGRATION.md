# Replay Integration

This document explains how traces integrate with [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester) for replay testing against the live Claude API.

## Overview

Traces capture the **structure** of real Claude Code conversations — token counts, timing, cache block patterns — without storing actual message content. A replay tool reconstructs realistic API requests that match these structural properties, then measures actual cache behavior on live infrastructure.

The Claude API is **stateless**: each request contains the entire conversation history. Messages accumulate: 1 → 3 → 5 → 7 → ...

## Conversation Accumulation

### How Messages Grow

Each turn adds 2 messages:
1. **Assistant response** (from previous API call)
2. **User message** (new text input or tool_result)

```
Turn 1: messages = [user_text]                                    → 1 message
Turn 2: messages = [user_text, assistant{...}, user{tool_result}] → 3 messages
Turn 3: messages = [user_text, asst{...}, user{...}, asst{...}, user{...}] → 5 messages
```

### Token Accumulation

Output tokens become input tokens in the next request:

```
Request N:   in=50000, out=500
Request N+1: in ≈ 50000 + 500 + tool_result_tokens + ~10 overhead
```

## Replaying Requests

### Following the Trace

Each trace request specifies `type`, `model`, `in`, `out`, `stop`, and timing. Replay sends matching API requests:

```python
def replay_request(req, messages, tools, system):
    is_streaming = req['type'] == 's'
    response = client.messages.create(
        model=req['model'],
        max_tokens=32000 if is_streaming else 21333,
        stream=is_streaming,
        system=system,
        tools=tools,
        messages=messages
    )
    return response
```

### Request Types

Trace requests have `type: "s"` (streaming) or `type: "n"` (non-streaming) depending on the proxy version used during collection. Both are functionally equivalent for replay — each represents one conversation turn.

## Tool Flow

When `stop: "tool_use"`, the next cross-turn request must include a `tool_result`:

```
Request N:   stop="tool_use", output_types=["thinking", "text", "tool_use"]
Request N+1: input_types=["tool_result"]
```

### Verification

```python
def verify_tool_flow(requests):
    expecting_result = False
    for req in requests:
        if req.get('type') == 'subagent':
            continue
        if 'tool_result' in req.get('input_types', []):
            if expecting_result:
                expecting_result = False  # Matched
        if req.get('stop') == 'tool_use':
            expecting_result = True
        elif req.get('stop') in ['', 'end_turn']:
            expecting_result = False
```

## Sub-Agent Handling

Sub-agents are nested conversations with independent caches:

```python
def replay_with_subagents(trace):
    parent_cache = {}
    for req in trace['requests']:
        if req.get('type') == 'subagent':
            # Fresh cache — sub-agents don't inherit parent cache
            subagent_cache = {}
            for sub_req in req.get('requests', []):
                replay_request(sub_req, subagent_cache)
            # subagent_cache discarded after completion
        else:
            replay_request(req, parent_cache)
```

Key differences from parent requests:
- **Independent cache** — no shared state with parent
- **Different tools/system** — often smaller tool sets (~8K vs ~12K tokens)
- **Different models** — often `claude-haiku` for speed
- **Relative timestamps** — sub-agent `t` values are relative to sub-agent start

## Handling Token Mismatches

The trace says request N had `out: 500`, but your API call returns 350 tokens. Now the next request's `in` won't match.

### Option 1: Pad to Target

```python
def pad_to_target(current_tokens, target_tokens, messages):
    deficit = target_tokens - current_tokens
    if deficit <= 0:
        return messages
    # Extend last text block with padding content
    padding = generate_padding_text(deficit)
    # ... append to last assistant or tool_result text
    return messages
```

### Option 2: Accept Drift

For simpler replay, accept that token counts will diverge from the trace. Cache behavior remains similar as long as the prefix content is preserved.

```python
drift_pct = abs(actual_in - expected_in) / expected_in * 100
if drift_pct > 20:
    print(f"Warning: {drift_pct:.1f}% drift from trace")
```

Cache operates on 64-token blocks, so mismatches smaller than 64 tokens have minimal effect.

## Expected Cache Behavior

When replaying correctly:

| Scenario | Expected Cache Hit Rate |
|----------|------------------------|
| Cross-turn (new content appended) | 93-99% of prefix |
| First request | 0% (or high if global cache warm from other sessions) |
| Sub-agent first request | 0% (independent cache context) |

The ~1-2pp gap between simulation and actual API is due to cross-conversation caching of the tool definitions + system prompt prefix.

## Validation

### Token Growth

```python
def validate_token_growth(requests, tolerance=0.15):
    prev = None
    for req in requests:
        if req.get('type') == 'subagent':
            continue
        if prev:
            expected = prev['in'] + prev['out']
            if not (expected <= req['in'] <= expected * (1 + tolerance)):
                print(f"Unexpected growth: {prev['in']}+{prev['out']} → {req['in']}")
        prev = req
```

### Cache Accuracy

Use `validate_trace_cache.py` to compare simulated cache hits (from hash_ids) against actual API metrics:

```bash
python3 validate_trace_cache.py traces/ --db requests.db --jsonl-dir jsonl/
```

Output shows per-trace simulated vs API cache rates:
```
02b62262-215 (1,216 reqs, 605 matched)
  Simulated: 98.5% (1,788,763/1,815,770 blocks)
  API:       99.4% (57,842,077/58,200,875 tokens)
  Accuracy:  99.1%
```
