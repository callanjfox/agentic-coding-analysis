# KV Cache Analysis

Tools for analyzing and simulating Claude's prompt caching behavior. Achieves **95-97% accuracy** against actual API cache metrics.

This project was created as part of my product management work at [WEKA](https://www.weka.io/) on the [Augmented Memory Grid](https://www.weka.io/resources/solution-brief/weka-augmented-memory-grid/) product that I lead. I needed to understand how Claude Code's KV cache behaves in real-world conversations — specifically how cache hit rates evolve over time, how TTL affects cache reuse, and what the working set looks like across different conversation patterns. The visualization tools (`complete_cache_visualizer.py`) were built to answer these questions, and the trace generation pipeline packages this data for replay testing on real storage infrastructure.

Generates compact **trace files** that capture cache block patterns from real Claude Code sessions. These traces feed into [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester) for replay testing against live infrastructure.

## Proxy Compatibility

The analysis tools work with data from both old and new versions of [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy):

- **Fixed proxy** (after [seifghazi/claude-code-proxy#33](https://github.com/seifghazi/claude-code-proxy/pull/33)): The database contains one streaming request per turn with complete metadata. The tools process these directly via JSONL message ID linking.
- **Older proxy**: A bug caused Claude Code to fire a non-streaming replay for every turn (the proxy stripped SSE headers, so Claude Code retried to recover metadata). Older databases contain paired streaming + non-streaming requests per turn. The tools use the non-streaming requests (which have complete `stop_reason`, content types, and usage data) and skip the streaming artifacts.

No configuration needed — the tools detect which format they're working with automatically.

## Data Sources

**Currently supported:** Two complementary sources used together:
1. **`requests.db`** from [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy) — a SQLite database capturing every API request/response body
2. **JSONL conversation files** from Claude Code's local storage (`~/.claude/projects/`) — provides conversation structure and message ID linking

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for details on how these two sources work together.

**Planned:** Claude Enterprise admin log console (direct API log ingestion without the proxy).

## Example Data

The `examples/` directory contains a complete working example: one Claude Code conversation where the user asked:

> *"Can you clone a version of vllm locally and then fully analyze its full code path and write me a markdown file about key components."*

This single prompt generated **92 API requests** across 4 sub-agents in ~17 minutes, processing **~3.8 million tokens** with a high cache hit rate.

| Metric | Value |
|--------|-------|
| Total requests (streaming only) | 92 |
| Models | claude-opus-4-6, claude-haiku-4-5 |
| Duration | ~17 minutes |
| Sub-agents spawned | 4 (Explore agents) |

See [docs/EXAMPLE_CONVERSATION.md](docs/EXAMPLE_CONVERSATION.md) for a detailed request-by-request walkthrough of this conversation — showing how the cache builds up, how sub-agents share their tool prefix, and why cache rates change at each phase.

**Included files:**
- `examples/requests.db` — Full API request/response database from [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy)
- `examples/jsonl/` — JSONL conversation files from Claude Code's local storage (1 parent + 4 sub-agent files)
- `examples/visualizations/` — Pre-generated interactive HTML charts (open in browser):
  - `combined_combined.html` — Combined dashboard with all cache metrics
  - `combined_ttl.html` — TTL impact analysis across different cache lifetimes
  - `combined_working_set.html` — Working set size over time
  - `combined_stats.txt` — Full text statistics

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Try it with the example data

```bash
# 1. Setup: build indexes linking JSONL to DB
python3 build_message_index.py examples/requests.db
python3 build_conversation_index.py examples/requests.db --projects-path examples/jsonl/

# 2. Generate traces with sub-agents
python3 build_minimal_traces.py examples/requests.db \
    --jsonl-dir examples/jsonl/ \
    --output-dir examples/traces/ \
    --block-size 64 \
    --include-subagents

# 3. Validate traces against actual API cache metrics
python3 validate_trace_cache.py examples/traces/ \
    --db examples/requests.db \
    --jsonl-dir examples/jsonl/
```

Expected output from validation:
```
02f3098e-6bc (16 reqs, 6 matched)
  Simulated: 85.8% (5,681/6,625 blocks)
  API:       94.3% (210,961/223,791 tokens)
  Accuracy:  91.0%
```

### Full workflow with your own data

#### Setup (one-time per database)

```bash
python3 build_message_index.py requests.db
python3 build_conversation_index.py requests.db
```

#### Analyze a Conversation

```bash
# Interactive HTML visualization
python3 complete_cache_visualizer.py requests.db --conversation-id <uuid> --output viz.html

# Text-only stats (no charts, faster)
python3 complete_cache_visualizer.py requests.db --conversation-id <uuid> --output stats.html --text-only

# List available conversations
python3 list_conversations.py requests.db
```

#### Generate Traces

```bash
# Generate traces from all conversations
python3 build_minimal_traces.py requests.db \
    --jsonl-dir jsonl/ \
    --output-dir traces/ \
    --block-size 64 \
    --min-requests 5 \
    --include-subagents

# Split long conversations at 12-hour gaps
python3 build_minimal_traces.py requests.db \
    --jsonl-dir jsonl/ \
    --output-dir traces/ \
    --block-size 64 \
    --split-at-gap 43200 \
    --include-subagents
```

#### Validate Traces Against API Metrics

```bash
python3 validate_trace_cache.py traces/ \
    --db requests.db \
    --jsonl-dir jsonl/
```

#### Recover Conversations (when JSONL files are missing)

```bash
# IMPORTANT: Always use --exclude-indexed to avoid duplicating
# requests already covered by real JSONL files
python3 recover_conversations.py requests.db \
    --output-dir recovered_jsonl/ \
    --exclude-indexed jsonl/ \
    --workers 16
```

## Scripts

| Script | Purpose |
|--------|---------|
| `build_minimal_traces.py` | Generate compact traces with hash_ids for cache replay |
| `validate_trace_cache.py` | Validate trace cache simulation against actual API metrics |
| `recover_conversations.py` | Recover conversation structure from DB when JSONL missing |
| `complete_cache_visualizer.py` | Cache analysis with interactive HTML charts (use `--text-only` for stats only) |
| `complete_conversation_builder.py` | Build complete request timelines |
| `build_message_index.py` | One-time setup: extract message IDs from responses |
| `build_conversation_index.py` | One-time setup: link JSONL conversations to DB requests |
| `list_conversations.py` | List available conversations in a database |
| `kv_common.py` | Shared utilities (tokenization, normalization, classification) |

## Trace Format

Traces are compact JSON files capturing cache block structure without actual message content. See [docs/TRACE_FORMAT.md](docs/TRACE_FORMAT.md) for the full specification.

```json
{
  "id": "8712b46f-04e",
  "models": ["claude-sonnet-4-20250514"],
  "block_size": 64,
  "hash_id_scope": "global",
  "tool_tokens": 11880,
  "system_tokens": 3427,
  "requests": [
    {
      "t": 0.0,
      "type": "s",
      "model": "claude-sonnet-4-20250514",
      "in": 19105,
      "out": 297,
      "hash_ids": [1, 2, 3, "...", 298],
      "input_types": ["text"],
      "output_types": ["thinking", "text", "tool_use"],
      "stop": "tool_use",
      "api_time": 5.81,
      "ttft": 3.52,
      "think_time": 0.0
    }
  ]
}
```

Key fields:
- **`hash_ids`** — Ordered block hashes for prefix-based cache matching. Two requests sharing a hash_id prefix share cached content.
- **`hash_id_scope`** — `"global"` means hash_ids are consistent across all conversations and sub-agents in a batch. Same content at the same position always gets the same ID, enabling cross-conversation cache simulation.
- **`tool_tokens` / `system_tokens`** — Shared prefix (~15K tokens) that stays warm in API's global cache across sessions.
- **`api_time`** — Total response time in seconds (from proxy's `responseTime`). How long the API took to respond.
- **`ttft`** — Time to first token in seconds (from `Server-Timing` header). Captures server-side latency before the first token.
- **`think_time`** — Client delay before this request in seconds. The gap between the previous response completing and this request being sent. Captures tool execution time, user reading time, and sub-agent wait time.

These timing fields allow the [trace replay tester](https://github.com/callanjfox/kv-cache-tester) to simulate different server speeds while preserving real client behavior. For example, `--timing-strategy api-scaled --api-time-scale 0.2` replays with a simulated 5x faster server but keeps real user/tool delays intact.

## How Caching Works

1. **Tokenize** request content in order: tools → system → messages
2. **Normalize** — remove `cache_control` and `signature` fields (they vary between requests)
3. **Block** — split into 64-token blocks (full blocks only; partial blocks are discarded)
4. **Chain hash** — each block's SHA256 hash includes the previous block's hash (encoding position) and a random salt generated once per run
5. **Assign IDs** — each unique hash gets a sequential integer ID

The salt prevents someone from tokenizing known content and confirming whether it appears in an anonymized trace. It is never written to the output files — the replay tool only compares pre-computed hash_ids and never needs to recompute them.

Cache hits are prefix-based: if block N misses, all subsequent blocks also miss. This means cache hit rate is determined by the length of the matching prefix.

### What the Simulation Does and Doesn't Cover

The cache simulation models **within-conversation prefix reuse** — as each turn appends new messages, the shared prefix from previous turns produces cache hits. It does **not** simulate cross-conversation caching of tool definitions and system prompts. In the real API, these shared prefixes (~15K tokens) are almost always warm from other active Claude Code sessions, giving even the first request of a new conversation a high cache hit rate. Our simulation treats the first request as a cold start.

### Why ~1.9% Gap Between Simulation and API

Three factors contribute to the remaining gap:

1. **Cross-conversation global cache** (~1-2pp) — The API's global cache keeps tool definitions (~12K tokens) and system prompt (~3K tokens) warm across all active sessions. Our simulation only tracks cache within a single conversation.

2. **Tokenizer differences** — We use tiktoken's GPT-4 tokenizer as an approximation. Claude's actual tokenizer produces slightly different token boundaries, which can cause minor mismatches in block alignment.

3. **Full-block rounding** — We only simulate full 64-token blocks, discarding the partial remainder (0-63 tokens per request). The API caches at a finer granularity, so our simulation systematically undercounts by a small amount.

## Integration with kv-cache-tester

Traces are consumed by [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester)'s `trace_replay_tester.py`, which replays them against the live Claude API to measure actual cache behavior on real infrastructure.

See [docs/REPLAY_INTEGRATION.md](docs/REPLAY_INTEGRATION.md) for details on conversation reconstruction, token flow, and validation.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/EXAMPLE_CONVERSATION.md](docs/EXAMPLE_CONVERSATION.md) | Step-by-step walkthrough of the example conversation showing how requests, cache, and sub-agents work |
| [docs/TRACE_FORMAT.md](docs/TRACE_FORMAT.md) | Complete trace JSON schema, hash algorithm, sub-agent format |
| [docs/REPLAY_INTEGRATION.md](docs/REPLAY_INTEGRATION.md) | How to replay traces, conversation accumulation, validation |
| [docs/CACHE_BEHAVIOR.md](docs/CACHE_BEHAVIOR.md) | Claude API caching mechanics, request pairing, TTL behavior |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | requests.db schema, JSONL files, recovery workflow, future plans |

## Validated Findings

| Metric | Value |
|--------|-------|
| Simulated cache hit rate (infinite TTL, within-conversation only) | ~97.7% avg |
| API actual cache hit rate (includes cross-conversation system/tool caching) | ~99.6% avg |
| Gap | ~1.9pp |

## Future Work

- **Global hash_ids** — Share hash-to-ID mapping across conversations so traces have common IDs for shared prefixes, enabling cross-conversation cache simulation
- **Absolute timestamps** — Optional ISO 8601 timestamps alongside relative seconds
- **Claude Enterprise log console** — Direct ingestion from enterprise admin logs without requiring the proxy
