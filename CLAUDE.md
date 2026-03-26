# CLAUDE.md

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
    --split-at-gap 43200
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

## Architecture

### Cache Simulation
- Block size: 64 tokens (default)
- Tokenizer: tiktoken GPT-4 encoder
- Hash: SHA256 chained (each block depends on previous)
- Normalization: removes `cache_control` and `signature` fields

### Two Requests Per Turn
Claude Code sends streaming (`max_tokens=32000`) then non-streaming (`max_tokens=21333`) with identical content. Non-streaming gets ~100% cache hit.

### Key Accuracy Numbers
- Simulation: ~97.7% cache hit rate (infinite TTL)
- API actual: ~99.6%
- Gap: ~1.9pp (cross-conversation global cache)

## Dependencies

`tiktoken`, `plotly`, `numpy` (see requirements.txt)
