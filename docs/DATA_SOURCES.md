# Data Sources

Trace generation requires **two complementary data sources** working together:

1. **`requests.db`** — from [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy), captures every API request and response
2. **JSONL files** — from Claude Code's local storage (`~/.claude/projects/`), provides conversation structure and message linking

The proxy captures the raw API traffic. The JSONL files provide the conversation thread structure that links requests into coherent sessions. Together, they enable accurate trace generation.

## The Two Sources in Detail

### 1. JSONL Conversation Files (from Claude Code client)

Claude Code (the CLI tool) saves conversation history to local JSONL files at `~/.claude/projects/<project-hash>/<conversation-id>.jsonl`.

**What they contain:**
- Complete message history as the user experiences it
- Message IDs (`msg_01XYZ...`) from assistant responses
- Tool use/result pairs with content
- Sub-agent conversations in separate `agent-<id>.jsonl` files
- Metadata: timestamps, `isSidechain` flag, `toolUseResult` with agent linking info

**What they capture:**
- The canonical response for each turn (one entry per assistant message)
- Message IDs link directly to the corresponding request in `requests.db`

**Example JSONL entry:**
```json
{"type": "assistant", "message": {"id": "msg_01ABC...", "content": [...]}, "timestamp": "2025-01-15T14:30:22Z"}
```

### 2. requests.db (from Claude Code Proxy)

The [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy) sits between Claude Code and the Anthropic API, capturing **every** request and response.

**What it contains:**
- SQLite database with a `requests` table
- Full request body (tools, system prompt, messages array)
- Full response body (content, usage metrics with cache breakdown)
- Timestamps for each request

**What it captures:**
- All API requests and responses
- Actual API cache metrics (`cache_read_input_tokens`, `cache_creation_input_tokens`)
- Token counts, model IDs, stop reasons
- Note: older proxy versions produced duplicate streaming/non-streaming pairs per turn due to a bug (see [seifghazi/claude-code-proxy#33](https://github.com/seifghazi/claude-code-proxy/pull/33)); the analysis tools handle both old and fixed proxy data

**Schema:**
```sql
CREATE TABLE requests (
    id INTEGER PRIMARY KEY,
    timestamp TEXT,
    body TEXT,      -- Full JSON request body
    response TEXT   -- Full JSON response body
);
```

## How They Link Together

The key linking mechanism is the **message ID** (`msg_01XYZ...`):

```
JSONL file                          requests.db
─────────                           ───────────
{"type": "assistant",               response.body.id = "msg_01ABC..."
 "message": {"id": "msg_01ABC..."}}
         │                                    │
         └──────────── MATCH ─────────────────┘
```

### Linking Pipeline

1. **`build_message_index.py`** scans `requests.db` and extracts `response.body.id` from every response → creates `message_id_index.json`

2. **`build_conversation_index.py`** scans JSONL files, extracts message IDs, and matches them against the index → creates `conversation_index` table in the database

3. **`build_minimal_traces.py`** uses both sources:
   - JSONL provides conversation structure and message ordering
   - requests.db provides full request bodies for tokenization and hashing
   - Message IDs link them together with 98-100% accuracy

### Why Message ID Matching Matters

Earlier approaches used **timestamp matching** (find the DB request closest in time to each JSONL entry). This only achieved 5-43% accuracy due to clock drift and ambiguous matches.

Message ID matching achieves **98-100%** because each `msg_01XYZ...` ID is globally unique and appears in exactly one DB response.

### What Gets Indexed

The JSONL files record one response per turn. Each message ID maps to exactly one request in `requests.db`, which the trace builder uses directly. In older proxy data (with streaming duplicates), the JSONL message IDs point to the non-streaming requests, which have complete metadata (`stop_reason`, content types, usage).

## Recovery: When JSONL Files Are Missing

Sometimes JSONL files are unavailable (deleted, different machine, lost). The `recover_conversations.py` script reconstructs conversation structure from `requests.db` alone.

### How Recovery Works

1. **Classify** each request as streaming or non-streaming by the `stream` field
2. **Group** requests into turns — anchoring on streaming requests, with optional non-streaming partners matched by content hash within 30 seconds
3. **Chain** turns into conversations by hash prefix overlap — if request B's hash_ids share >97% prefix with request A, they're in the same conversation
4. **Order** by timestamp to reconstruct the conversation timeline

### Important: Use `--exclude-indexed`

When generating traces from both real JSONL and recovered data:

```bash
python3 recover_conversations.py requests.db \
    --output-dir recovered_jsonl/ \
    --exclude-indexed jsonl/ \
    --workers 16
```

The `--exclude-indexed` flag ensures recovered conversations only contain requests **not already covered** by real JSONL files. It works by:
1. Scanning all JSONL files in the specified directory
2. Extracting message IDs from assistant entries
3. Skipping any DB request whose response contains a matching message ID

This prevents duplicate requests appearing in both real and recovered traces.

## End-to-End Data Flow

### Setup: How Data Is Collected

```
                        Claude Code (CLI)
                              │
              ┌───────────────┼───────────────────┐
              │               │                   │
              ▼               │                   ▼
    JSONL conversation        │          claude-code-proxy
    files saved locally       │          (github.com/seifghazi/claude-code-proxy)
              │               │                   │
    ~/.claude/projects/       │                   ▼
    <hash>/<conv-id>.jsonl    │            requests.db
    (one entry per turn)       │            (ALL requests + responses)
                              │
                              ▼
                        Anthropic API
```

**The setup:** Claude Code is configured to route API calls through `claude-code-proxy`. The proxy saves every request/response to `requests.db` and forwards them to the Anthropic API. Separately, Claude Code saves its conversation state to local JSONL files as it normally does.

### Analysis Pipeline

```
    requests.db + JSONL files
        │
        ├─── build_message_index.py ──→ message_id_index.json
        ├─── build_conversation_index.py ──→ conversation_index table
        │
        ├─── complete_cache_visualizer.py ──→ HTML visualizations
        ├─── complete_cache_analyzer.py ──→ text statistics
        │
        └─── build_minimal_traces.py ──→ trace JSON files
                                              │
                                              └──→ kv-cache-tester (replay)
```

## Future: Claude Enterprise Admin Log Console

Currently, data collection requires running [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy) to capture requests, plus access to Claude Code's local JSONL files. A planned future enhancement will support ingesting data directly from the **Claude Enterprise admin log console**, which provides API request/response logs without needing a proxy.

### Planned Approach

The Enterprise log console provides similar data to requests.db (request bodies, response bodies, usage metrics) but through an admin API rather than a local proxy. The integration would:

1. Fetch request logs from the Enterprise API
2. Convert to the same internal format used by the analysis scripts
3. Generate traces using the existing `build_minimal_traces.py` pipeline

This would eliminate the proxy requirement for Enterprise customers and provide access to logs from any team member's Claude Code sessions.

### What Changes

| Aspect | Current (Proxy) | Future (Enterprise) |
|--------|----------------|-------------------|
| Data capture | Local proxy intercepts | Enterprise admin API |
| Scope | Single user's sessions | All team sessions |
| JSONL files | From local `~/.claude/` | Not available (DB-only recovery) |
| Setup | Install + configure proxy | Enterprise admin access |

The trace format and analysis tools remain unchanged — only the data ingestion layer differs.
