# Example Visualizations

Pre-generated interactive charts from `complete_cache_visualizer.py --conversation-id combined` analyzing all 184 requests in the example database.

## Charts

| File | Description |
|------|-------------|
| [combined_combined.html](combined_combined.html) | Combined dashboard — cache hit rates, token growth, streaming vs non-streaming breakdown |
| [combined_ttl.html](combined_ttl.html) | TTL impact — how cache lifetime (1 min to 24 hr) affects hit rate |
| [combined_working_set.html](combined_working_set.html) | Working set — active cache size over time |

Download and open in a browser, or view via GitHub Pages if enabled.

## Summary Statistics

From `combined_stats.txt`:

```
Total Requests:     184 (92 streaming, 92 non-streaming)
Models:             claude-opus-4-6 (13 reqs), claude-haiku-4-5 (171 reqs)
Total Input Tokens: 7,414,747
Cache Hit Rate:     94.7% (simulated, 5-min TTL)
API Actual Rate:    93.3%
Accuracy:           96.1%

Context Length:     median 38,684 tokens, max 88,834
Time Between Reqs:  median 1.0s, mean 4.6s
```

## Regenerate

```bash
python3 build_message_index.py examples/requests.db
python3 build_conversation_index.py examples/requests.db --projects-path examples/jsonl/

python3 complete_cache_visualizer.py examples/requests.db \
    --conversation-id combined \
    --output examples/visualizations/combined.html \
    --skip-long-ttls
```
