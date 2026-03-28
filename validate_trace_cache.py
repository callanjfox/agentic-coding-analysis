#!/usr/bin/env python3
"""
Validate Trace Cache Hit Rates Against API Metrics

Properly validates per-conversation cache hit rates by:
1. Loading trace file with conversation ID
2. Finding matching JSONL file
3. Extracting message_ids from JSONL assistant messages
4. Querying database for ONLY those message_ids
5. Comparing simulated vs actual API cache metrics

Usage:
    # Single trace
    python3 validate_trace_cache.py traces/locallinux/02b62262-215.json \
        --db request_dbs/requests_from_locallinux.db \
        --jsonl-dir jsonl/

    # All traces in directory
    python3 validate_trace_cache.py traces/locallinux/ \
        --db request_dbs/requests_from_locallinux.db \
        --jsonl-dir jsonl/
"""

import sqlite3
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def extract_message_ids_from_jsonl(jsonl_path: Path) -> List[str]:
    """Extract all message IDs from a JSONL conversation file"""
    message_ids = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                try:
                    msg = json.loads(line.strip())
                    if msg.get('type') == 'assistant':
                        message_obj = msg.get('message', {})
                        msg_id = message_obj.get('id')
                        if msg_id and msg_id.startswith('msg_'):
                            message_ids.append(msg_id)
                except:
                    pass
    except:
        pass
    return message_ids


def build_msg_id_to_usage_map(conn, message_ids: List[str]) -> Dict[str, Dict]:
    """Build map of message_id -> usage stats for specific message IDs"""
    if not message_ids:
        return {}

    cursor = conn.cursor()

    # Get all responses and filter by message_id
    cursor.execute("SELECT response FROM requests WHERE response IS NOT NULL")

    usage_map = {}
    for (resp_str,) in cursor.fetchall():
        try:
            resp = json.loads(resp_str)
            body = resp.get('body', resp)
            msg_id = body.get('id')

            if msg_id and msg_id in message_ids:
                usage = body.get('usage', {})
                usage_map[msg_id] = {
                    'input_tokens': usage.get('input_tokens', 0),
                    'cache_read': usage.get('cache_read_input_tokens', 0),
                    'cache_creation': usage.get('cache_creation_input_tokens', 0),
                    'output_tokens': usage.get('output_tokens', 0)
                }
        except:
            pass

    return usage_map


def _flatten_trace_requests(requests: list) -> list:
    """Flatten sub-agent nested requests into a single timeline."""
    result = []
    for req in requests:
        if req.get('type') == 'subagent':
            for sub_req in req.get('requests', []):
                result.append(sub_req)
        else:
            result.append(req)
    return sorted(result, key=lambda r: r.get('t', 0))


def simulate_cache_from_trace(trace: Dict, ttl_seconds: int = None) -> Tuple[int, int, float]:
    """Simulate cache from trace hash_ids.

    Args:
        trace: Trace data with requests containing hash_ids
        ttl_seconds: Cache TTL in seconds. None = infinite (no expiration)
    """
    total_blocks = 0
    cache_hits = 0

    # For global hash_id scope, flatten sub-agent requests into timeline
    # (their hash_ids are consistent with parent)
    scope = trace.get('hash_id_scope', 'per_context')
    if scope == 'global':
        requests = _flatten_trace_requests(trace.get('requests', []))
    else:
        # Legacy: skip sub-agent requests (hash_ids would collide)
        requests = [r for r in trace.get('requests', []) if r.get('type') != 'subagent']

    if ttl_seconds is None:
        # Infinite TTL - simple set-based tracking
        cache = set()
        for req in requests:
            hash_ids = req.get('hash_ids', [])
            hits = sum(1 for h in hash_ids if h in cache)
            cache.update(hash_ids)
            total_blocks += len(hash_ids)
            cache_hits += hits
    else:
        # Finite TTL - track expiry times
        cache = {}  # hash_id -> expiry_time
        for req in requests:
            t = req.get('t', 0)
            hash_ids = req.get('hash_ids', [])
            hits = sum(1 for h in hash_ids if h in cache and cache[h] > t)
            for h in hash_ids:
                cache[h] = t + ttl_seconds
            total_blocks += len(hash_ids)
            cache_hits += hits

    hit_rate = (cache_hits / total_blocks * 100) if total_blocks > 0 else 0
    return total_blocks, cache_hits, hit_rate


def find_jsonl_file(jsonl_dir: Path, conversation_id: str) -> Optional[Path]:
    """Find JSONL file matching conversation ID"""
    # Try exact match first
    exact = jsonl_dir / f"{conversation_id}.jsonl"
    if exact.exists():
        return exact

    # Try prefix match (trace id is first 12 chars)
    for jf in jsonl_dir.glob("*.jsonl"):
        if jf.stem.startswith(conversation_id[:12]):
            return jf
        if conversation_id.startswith(jf.stem[:12]):
            return jf

    return None


def validate_trace(trace_path: Path, db_path: str, jsonl_dir: Path, ttl_seconds: int = 300) -> Dict:
    """Validate a single trace against API metrics"""

    # Load trace
    with open(trace_path) as f:
        trace = json.load(f)

    trace_id = trace.get('id', trace_path.stem)
    num_requests = len(trace.get('requests', []))

    # Simulate cache from hash_ids
    total_blocks, cache_hits, sim_rate = simulate_cache_from_trace(trace, ttl_seconds)

    # Find matching JSONL
    jsonl_path = find_jsonl_file(jsonl_dir, trace_id)
    if not jsonl_path:
        return {
            'trace_id': trace_id,
            'requests': num_requests,
            'sim_blocks': total_blocks,
            'sim_hits': cache_hits,
            'sim_rate': sim_rate,
            'api_matched': 0,
            'api_total': 0,
            'api_cache_read': 0,
            'api_rate': None,
            'error': f"No JSONL file found for {trace_id}"
        }

    # Extract message IDs from JSONL
    message_ids = extract_message_ids_from_jsonl(jsonl_path)
    if not message_ids:
        return {
            'trace_id': trace_id,
            'requests': num_requests,
            'sim_blocks': total_blocks,
            'sim_hits': cache_hits,
            'sim_rate': sim_rate,
            'api_matched': 0,
            'api_total': 0,
            'api_cache_read': 0,
            'api_rate': None,
            'error': f"No message IDs in JSONL {jsonl_path.name}"
        }

    # Query database for these specific message IDs
    conn = sqlite3.connect(db_path)
    usage_map = build_msg_id_to_usage_map(conn, message_ids)
    conn.close()

    if not usage_map:
        return {
            'trace_id': trace_id,
            'requests': num_requests,
            'sim_blocks': total_blocks,
            'sim_hits': cache_hits,
            'sim_rate': sim_rate,
            'api_matched': 0,
            'api_total': 0,
            'api_cache_read': 0,
            'api_rate': None,
            'error': f"No matching requests in database for {len(message_ids)} message IDs"
        }

    # Calculate per-conversation API metrics
    api_input = sum(u['input_tokens'] for u in usage_map.values())
    api_cache_read = sum(u['cache_read'] for u in usage_map.values())
    api_cache_creation = sum(u['cache_creation'] for u in usage_map.values())
    api_total = api_input + api_cache_read + api_cache_creation

    api_rate = (api_cache_read / api_total * 100) if api_total > 0 else 0

    return {
        'trace_id': trace_id,
        'requests': num_requests,
        'sim_blocks': total_blocks,
        'sim_hits': cache_hits,
        'sim_rate': sim_rate,
        'api_matched': len(usage_map),
        'api_total': api_total,
        'api_cache_read': api_cache_read,
        'api_rate': api_rate,
        'error': None
    }


def main():
    parser = argparse.ArgumentParser(description='Validate trace cache hit rates against API metrics')
    parser.add_argument('trace_path', help='Path to trace file or directory')
    parser.add_argument('--db', required=True, help='Path to requests.db')
    parser.add_argument('--jsonl-dir', required=True, help='Directory containing JSONL files')
    parser.add_argument('--ttl', type=int, default=None, help='Cache TTL in seconds (default: None = infinite)')
    args = parser.parse_args()

    trace_path = Path(args.trace_path)
    jsonl_dir = Path(args.jsonl_dir)

    if not jsonl_dir.exists():
        print(f"Error: JSONL directory not found: {jsonl_dir}")
        return

    # Get list of traces to validate
    if trace_path.is_file():
        traces = [trace_path]
    elif trace_path.is_dir():
        traces = sorted(trace_path.glob("*.json"))
    else:
        print(f"Error: Path not found: {trace_path}")
        return

    print(f"Validating {len(traces)} trace(s) against API metrics")
    print(f"Database: {args.db}")
    print(f"JSONL dir: {jsonl_dir}")
    print(f"TTL: {'infinite' if args.ttl is None else f'{args.ttl}s'}")
    print("=" * 70)
    print()

    results = []
    for tp in traces:
        result = validate_trace(tp, args.db, jsonl_dir, args.ttl)
        results.append(result)

        if result['error']:
            print(f"{result['trace_id']}: ERROR - {result['error']}")
        else:
            accuracy = (result['sim_rate'] / result['api_rate'] * 100) if result['api_rate'] else 0
            print(f"{result['trace_id']} ({result['requests']:,} reqs, {result['api_matched']:,} matched)")
            print(f"  Simulated: {result['sim_rate']:.1f}% ({result['sim_hits']:,}/{result['sim_blocks']:,} blocks)")
            print(f"  API:       {result['api_rate']:.1f}% ({result['api_cache_read']:,}/{result['api_total']:,} tokens)")
            print(f"  Accuracy:  {accuracy:.1f}%")
            print()

    # Summary
    if len(results) > 1:
        valid_results = [r for r in results if r['api_rate'] is not None]
        if valid_results:
            print("=" * 70)
            print("SUMMARY")
            print("-" * 70)
            api_rates = [r['api_rate'] for r in valid_results]
            sim_rates = [r['sim_rate'] for r in valid_results]
            print(f"Valid traces: {len(valid_results)}/{len(results)}")
            print(f"API rate range: {min(api_rates):.1f}% - {max(api_rates):.1f}%")
            print(f"Sim rate range: {min(sim_rates):.1f}% - {max(sim_rates):.1f}%")

            # Check if API rates vary (the bug fix)
            if max(api_rates) - min(api_rates) < 0.5:
                print()
                print("WARNING: API rates are nearly identical - possible bug!")
            else:
                print()
                print("API rates vary as expected - per-conversation validation working.")


if __name__ == "__main__":
    main()
