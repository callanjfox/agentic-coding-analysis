# KV Cache Analysis

Tools for analyzing and simulating Claude's prompt caching behavior. Achieves **95-97% accuracy** against actual API cache metrics.

Generates compact **trace files** that capture cache block patterns from real Claude Code sessions. These traces feed into [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester) for replay testing against live infrastructure.

## Key Discovery

Claude Code sends **two requests per turn**:
- **Streaming** (`max_tokens=32000`) — for real-time UI responsiveness
- **Non-streaming** (`max_tokens=21333`) — for tool execution, 5-8 seconds later

Both have identical content. The non-streaming request gets ~100% cache hit from the streaming request's cache creation. This means half of all API traffic is essentially free.

## Data Sources

**Currently supported:** Two complementary sources used together:
1. **`requests.db`** from [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy) — a SQLite database capturing every API request/response body (both streaming and non-streaming)
2. **JSONL conversation files** from Claude Code's local storage (`~/.claude/projects/`) — provides conversation structure and message ID linking

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for details on how these two sources work together.

**Planned:** Claude Enterprise admin log console (direct API log ingestion without the proxy).

## Example Data

The `examples/` directory contains a complete working example: one Claude Code conversation where the user asked:

> *"Can you clone a version of vllm locally and then fully analyze its full code path and write me a markdown file about key components."*

This single prompt generated **185 API requests** across 4 sub-agents in ~17 minutes, processing **7.7 million tokens** with a 93.3% cache hit rate.

| Metric | Value |
|--------|-------|
| Total requests | 185 |
| Streaming / Non-streaming | 92 / 93 |
| Models | claude-opus-4-6, claude-haiku-4-5 |
| Duration | ~17 minutes |
| Sub-agents spawned | 4 (Explore agents) |
| Total tokens processed | 7.7M |
| Cache hit rate | 93.3% |
| Cache read tokens | 7.2M |

**Included files:**
- `examples/requests.db` — Full API request/response database from [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy)
- `examples/jsonl/` — JSONL conversation files from Claude Code's local storage (1 parent + 4 sub-agent files)

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

# Text-only analysis (faster)
python3 complete_cache_analyzer.py requests.db --conversation-id <uuid>

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
| `complete_cache_visualizer.py` | Interactive HTML cache behavior visualizations |
| `complete_cache_analyzer.py` | Text-only cache analysis (no plotly dependency) |
| `complete_conversation_builder.py` | Build complete request timelines with streaming pair matching |
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
      "stop": "tool_use"
    }
  ]
}
```

Key fields:
- **`hash_ids`** — Ordered block hashes for prefix-based cache matching. Two requests sharing a hash_id prefix share cached content.
- **`tool_tokens` / `system_tokens`** — Shared prefix (~15K tokens) that stays warm in API's global cache across sessions.

## How Caching Works

1. **Tokenize** request content in order: tools → system → messages
2. **Normalize** — remove `cache_control` and `signature` fields (they vary between requests)
3. **Block** — split into 64-token blocks (full blocks only; partial blocks are discarded)
4. **Chain hash** — each block's SHA256 hash includes the previous block's hash, encoding position
5. **Assign IDs** — each unique hash gets a sequential integer ID

Cache hits are prefix-based: if block N misses, all subsequent blocks also miss. This means cache hit rate is determined by the length of the matching prefix.

### Why ~1.9% Gap Between Simulation and API

Our simulation is correct for within-conversation prefix matching. The API achieves ~1.9 percentage points higher because it has a **global cache** shared across all Claude Code sessions. Tool definitions (~12K tokens) and system prompt (~3K tokens) are always warm from other active sessions, so even the first request of a new conversation gets ~100% cache hit.

## Integration with kv-cache-tester

Traces are consumed by [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester)'s `trace_replay_tester.py`, which replays them against the live Claude API to measure actual cache behavior on real infrastructure.

See [docs/REPLAY_INTEGRATION.md](docs/REPLAY_INTEGRATION.md) for details on conversation reconstruction, token flow, and validation.

## Documentation

| Document | Contents |
|----------|----------|
| [docs/TRACE_FORMAT.md](docs/TRACE_FORMAT.md) | Complete trace JSON schema, hash algorithm, sub-agent format |
| [docs/REPLAY_INTEGRATION.md](docs/REPLAY_INTEGRATION.md) | How to replay traces, conversation accumulation, validation |
| [docs/CACHE_BEHAVIOR.md](docs/CACHE_BEHAVIOR.md) | Claude API caching mechanics, request pairing, TTL behavior |
| [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) | requests.db schema, JSONL files, recovery workflow, future plans |

## Validated Findings

| Metric | Value |
|--------|-------|
| Simulated cache hit rate (infinite TTL) | ~97.7% avg |
| API actual cache hit rate | ~99.6% avg |
| Gap (cross-conversation caching) | ~1.9pp |
| Within-turn pair cache reuse | 100% |
| Non-streaming cache hit rate | 97-99% |

## Future Work

- **Global hash_ids** — Share hash-to-ID mapping across conversations so traces have common IDs for shared prefixes, enabling cross-conversation cache simulation
- **Absolute timestamps** — Optional ISO 8601 timestamps alongside relative seconds
- **Claude Enterprise log console** — Direct ingestion from enterprise admin logs without requiring the proxy
