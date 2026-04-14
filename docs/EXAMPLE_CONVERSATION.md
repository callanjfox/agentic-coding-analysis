# Example Conversation Walkthrough

This walks through the included example database (`examples/requests.db`) to show how a real Claude Code conversation generates API requests and how the KV cache behaves.

> **Note on streaming duplicates:** Older `requests.db` files from [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy) contain paired streaming + non-streaming requests per turn — the proxy's broken SSE forwarding caused Claude Code to replay each turn as a non-streaming request. The streaming requests are artifacts with incomplete metadata. This example database has been cleaned to contain only the non-streaming requests (which have complete `stop_reason`, content types, and usage data). See [seifghazi/claude-code-proxy#33](https://github.com/seifghazi/claude-code-proxy/pull/33) for the proxy fix.

## The Prompt

> "Can you clone a version of vllm locally and then fully analyze its full code path and write me a markdown file about key components."

This single prompt generated **~92 streaming API requests** over **~17 minutes**. The parent conversation (claude-opus-4-6) spawned 4 sub-agents (claude-haiku-4-5) to explore the codebase in parallel.

## Phase 1: Parent Initial Request

```
  STREAM  opus  25,407 tokens in →  205 out | cache: 83.6% | api: 5.8s
```

The parent sends the user's prompt. The 25K input tokens are: tools (~15K) + system prompt (~6K) + user message.

The first request already shows 83.6% cache hit because the API's global cache keeps tool definitions warm from other sessions.

## Phase 2: Parent Tool Calls

```
  STREAM  opus  25,627 tokens in →  123 out | cache: 99.1% | api: 8.6s | think: 4.0s
  STREAM  opus  25,830 tokens in →  822 out | cache: 99.2% | api: 14.8s | stop: tool_use
```

The parent clones the vLLM repo and inspects it. Context grows from 25K to 26K as tool results are added. Cache stays at **99%** -- only the new tool result tokens at the end need processing. The 4-second `think_time` is the client executing the tool (git clone) before sending the next request.

## Phase 3: 4 Sub-Agents Spawn Simultaneously

```
  STREAM  haiku 13,206 tokens in →  148 out | cache:  0.0% | api: 2.1s  <- SA1
  STREAM  haiku 13,249 tokens in →  101 out | cache:  0.0% | api: 1.5s  <- SA2
  STREAM  haiku 13,255 tokens in →  107 out | cache:  0.0% | api: 1.4s  <- SA3
  STREAM  haiku 13,215 tokens in →  109 out | cache:  0.0% | api: 1.6s  <- SA4
```

All 4 Explore agents fire within **the same second** (15:43:05). Their first requests all show **0% cache** -- not only because they use different tools than the parent, but because all 4 are in-flight simultaneously. The cache hasn't been written by any of them yet when the others arrive.

On the **second turn**, the sub-agents benefit from each other -- all 4 share identical tool definitions (~10K tokens), so the cache created by each sub-agent's first request is reusable by the others:

```
  STREAM  haiku 13,397 tokens in →   91 out | cache: 98.6%  <- SA1 turn 2
  STREAM  haiku 13,868 tokens in →   91 out | cache: 95.3%  <- SA4 turn 2
  STREAM  haiku 13,500 tokens in →   83 out | cache: 98.2%  <- SA3 turn 2
  STREAM  haiku 13,871 tokens in →   86 out | cache: 91.5%  <- SA2 turn 2
```

The ~10K shared tool prefix is now warm from turn 1, giving 91-98% cache hits.

## Phase 4: Sub-Agents Work Concurrently (~2 minutes)

~84 streaming requests from 4 sub-agents running in parallel, each exploring different parts of the vLLM codebase. The pattern is consistent:

- Context grows from ~13K to ~90K tokens as each agent reads files and accumulates conversation history
- Within each sub-agent: **93-100% cache hit** (prefix preserved, only new tool results added)
- Across sub-agents: shared tool prefix (~10K tokens) stays warm in global cache

```
  SA1: ~21 requests, grows to 73K tokens
  SA2: ~14 requests, grows to 89K tokens
  SA3: ~26 requests, grows to 83K tokens
  SA4: ~23 requests, grows to 73K tokens
```

## Phase 5: Parent Resumes

```
  STREAM  opus  44,604 tokens in → 6,421 out | cache: 57.9% | api: 120s
```

The parent resumes after all sub-agents complete. The cache hit drops to 58% -- but **not because the cache expired**. The breakdown:

| Component | Tokens | Status |
|-----------|--------|--------|
| Original prefix (tools + system + early messages) | 25,829 | **Cached** -- still warm from Phase 2 |
| Sub-agent results (new content injected) | 5,954 | **Created** -- never cached before |
| Request overhead | 12,821 | Uncached |

The context grew from 26K to 45K tokens because the sub-agent results were injected into the conversation. The old prefix is 100% cached; the "low" percentage is simply the ratio of old-to-new content.

By the next request, the new content is also cached:

```
  STREAM  opus  50,916 tokens in →   441 out | cache: 62.4% | api: 11s | stop: end_turn
```

The parent writes the architecture summary (441 output tokens) and the conversation ends with `end_turn`.

## Phase 6: User Follow-Up

```
  STREAM  opus  51,407 tokens in →  108 out | cache: 99.0% | api: 3.3s | think: 315s
```

The user reads the output for **~5 minutes** (`think_time: 315s`), then asks "Rewrite this in pirate speak." The cache is still warm -- 99% hit despite the long pause. The `think_time` field captures this real-world user behavior pattern.

## Timing Breakdown

Each request records three timing components:

| Metric | Range | What It Measures |
|--------|-------|------------------|
| `api_time` | 0.9s - 197s | Total response time (prefill + decode + streaming) |
| `ttft` | 0.6s - 3.5s | Time to first token -- server-side latency before first token |
| `think_time` | 0s - 315s | Client delay before sending request |

- **`ttft`**: Derived from the `Server-Timing` response header (or `X-Envoy-Upstream-Service-Time` for older proxy versions).
- **Tool call results**: `think_time = 0-4s` (tool execution time)
- **User reading**: `think_time = 315s` (user thinking/reading)
- **Sub-agent wait**: `think_time = 284s` (parent waiting for sub-agents)

These allow the [trace replay tester](https://github.com/callanjfox/kv-cache-tester) to simulate different server speeds while preserving real client behavior patterns.

## Cache Summary

| Scenario | Cache Hit Rate | Why |
|----------|---------------|-----|
| Within-conversation growth | 93-99% | Prefix preserved, only new suffix tokens |
| First sub-agent request (all 4 concurrent) | 0% | All in-flight simultaneously, cache not yet written |
| Sub-agent turn 2+ | 91-98% | Shared tool prefix warm from other sub-agents' turn 1 |
| Parent after sub-agents | 58% -> 100% | Old prefix cached, new results created, then warm |
| After 5-minute user pause | 99% | Cache lives beyond typical session TTL |

## How to Reproduce

```bash
# Setup
python3 build_message_index.py examples/requests.db
python3 build_conversation_index.py examples/requests.db --projects-path examples/jsonl/

# Generate trace with timing data (--anonymize strips IDs and salts hashes)
python3 build_minimal_traces.py examples/requests.db \
    --jsonl-dir examples/jsonl/ \
    --output-dir examples/traces/ \
    --block-size 64 \
    --include-subagents \
    --anonymize

# Validate against API metrics
python3 validate_trace_cache.py examples/traces/ \
    --db examples/requests.db \
    --jsonl-dir examples/jsonl/

# Visualize
python3 complete_cache_visualizer.py examples/requests.db \
    --conversation-id combined \
    --output examples/visualizations/combined.html
```
