# CLAUDE.md

## Repository Layout

This is the **agentic-coding-analysis** repo (`github.com/callanjfox/agentic-coding-analysis`).

The companion repo is **kv-cache-tester** (`github.com/callanjfox/kv-cache-tester`) — the trace replay tester that consumes traces generated here.

Local working directories:
```
/mnt/weka/kv-analysis/repos/agentic-coding-analysis/   # this repo
/mnt/weka/kv-analysis/repos/kv-cache-tester/            # companion repo
```

Generated trace data:
```
/mnt/weka/kv-analysis/traces_v6/anonymized/             # 270 traces (JSONL-based only)
/mnt/weka/kv-analysis/traces_v6/anonymized_recovered/   # 522 traces (JSONL + recovered)
```

## Project Overview

Tools for analyzing Claude's prompt caching behavior using request data from [claude-code-proxy](https://github.com/seifghazi/claude-code-proxy) and JSONL conversation files from Claude Code's local storage.

Generates compact **trace files** for replay testing via [kv-cache-tester](https://github.com/callanjfox/kv-cache-tester).

## Common Commands

### Setup (one-time per database)
```bash
python3 build_message_index.py requests.db
python3 build_conversation_index.py requests.db
```

### Cache Analysis
```bash
# HTML visualization
python3 complete_cache_visualizer.py requests.db --conversation-id <uuid> --output viz.html

# Batch analysis
python3 complete_cache_visualizer.py requests.db --conversation-id all --output-dir visualizations/

# Text-only stats (no charts, faster)
python3 complete_cache_visualizer.py requests.db --conversation-id <uuid> --text-only
```

### Trace Generation
```bash
python3 build_minimal_traces.py requests.db \
    --jsonl-dir jsonl/ \
    --output-dir traces/ \
    --block-size 64 \
    --min-requests 5 \
    --include-subagents \
    --anonymize
```

### Trace Validation
```bash
python3 validate_trace_cache.py traces/ --db requests.db --jsonl-dir jsonl/
```

### Conversation Recovery
```bash
python3 recover_conversations.py requests.db \
    --output-dir recovered_jsonl/ \
    --exclude-indexed jsonl/ \
    --workers 16
```

## Data Sources

**Current:** `requests.db` (from claude-code-proxy) + JSONL files (from `~/.claude/projects/`)
**Future:** Claude Enterprise admin log console

## Key Features

- **Global hash_ids**: Shared across all conversations in a batch — enables cross-conversation cache simulation
- **Timing breakdown**: `api_time` (total response time), `ttft` (time to first token from Server-Timing header), and `think_time` (client delay) per request
- **Sub-agent support**: Nested sub-agent traces with isolated cache contexts
- **Anonymization**: `--anonymize` strips conversation IDs, timestamps, and agent IDs

## Architecture

### Cache Simulation
- Block size: 64 tokens (default)
- Tokenizer: tiktoken GPT-4 encoder
- Hash: SHA256 chained (each block depends on previous), salted with a random value per run to prevent content identification in anonymized traces
- Normalization: removes `cache_control` and `signature` fields

### Two Requests Per Tool Call
Claude Code sends streaming then non-streaming with identical content. Non-streaming gets ~100% cache hit.

### Key Accuracy Numbers
- Simulation: ~95.4% cache hit rate (with global hash_ids, sub-agents included)
- API actual: ~94.3%
- Gap: eliminated with global hash_ids

## Dependencies

`tiktoken`, `plotly`, `numpy` (see requirements.txt)
