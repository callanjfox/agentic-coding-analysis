#!/usr/bin/env python3
"""
Complete Cache Analyzer

Analyzes cache behavior using COMPLETE conversation (streaming + non-streaming).
This provides accurate statistics matching actual API behavior.
"""

import sqlite3
import json
import hashlib
import tiktoken
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CacheChunk:
    """Cache chunk"""
    token_count: int
    chunk_hash: str
    sequence_number: int
    created_at: datetime
    last_accessed: datetime

    def is_expired(self, ts: datetime, ttl_seconds: int = 300) -> bool:
        return (ts - self.last_accessed).total_seconds() > ttl_seconds

    def refresh(self, ts: datetime):
        self.last_accessed = ts

    @staticmethod
    def create_from_tokens(token_ids: List[int], seq_num: int, ts: datetime):
        token_str = " ".join(map(str, token_ids))
        chunk_hash = hashlib.md5(token_str.encode()).hexdigest()[:16]
        return CacheChunk(
            token_count=len(token_ids),
            chunk_hash=chunk_hash,
            sequence_number=seq_num,
            created_at=ts,
            last_accessed=ts
        )

@dataclass
class RequestAnalysis:
    """Analysis for one request"""
    index: int
    request_id: str
    request_type: str
    message_count: int
    timestamp: datetime
    total_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    cache_hit_rate: float
    api_cache_read: Optional[int]
    api_cache_creation: Optional[int]
    api_input_tokens: Optional[int]

class CompleteCacheSimulator:
    """Cache simulator that processes ALL requests"""

    def __init__(self, block_size: int = 64, ttl: int = 300):
        self.block_size = block_size
        self.ttl = ttl
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.cache: Dict[str, Dict[int, CacheChunk]] = {}
        self.analyses: List[RequestAnalysis] = []

    def extract_all_tokens(self, body):
        """Extract all tokens from request"""
        all_tokens = []

        if 'tools' in body and body['tools']:
            all_tokens.extend(self.tokenizer.encode(json.dumps(body['tools'], separators=(',', ':'))))

        if 'system' in body:
            all_tokens.extend(self.tokenizer.encode(json.dumps(body['system'], separators=(',', ':'))))

        if 'messages' in body and body['messages']:
            all_tokens.extend(self.tokenizer.encode(json.dumps(body['messages'], separators=(',', ':'))))

        return all_tokens

    def chunk_tokens(self, tokens: List[int], ts: datetime) -> List[CacheChunk]:
        """Create chunks"""
        chunks = []
        for i in range(0, len(tokens), self.block_size):
            chunk_tokens = tokens[i:i + self.block_size]
            chunk = CacheChunk.create_from_tokens(chunk_tokens, i // self.block_size, ts)
            chunks.append(chunk)
        return chunks

    def find_prefix_match(self, chunks: List[CacheChunk], ts: datetime) -> Tuple[int, int]:
        """Find longest prefix match"""
        matched_tokens = 0
        matched_chunks = 0

        for chunk in chunks:
            if chunk.chunk_hash in self.cache:
                cached_chunks = self.cache[chunk.chunk_hash]
                if chunk.sequence_number in cached_chunks:
                    cached_chunk = cached_chunks[chunk.sequence_number]
                    if not cached_chunk.is_expired(ts, self.ttl):
                        cached_chunk.refresh(ts)
                        matched_tokens += chunk.token_count
                        matched_chunks += 1
                    else:
                        break
                else:
                    break
            else:
                break

        return matched_tokens, matched_chunks

    def update_cache(self, chunks: List[CacheChunk], start_from: int):
        """Update cache with new chunks"""
        for chunk in chunks[start_from:]:
            if chunk.chunk_hash not in self.cache:
                self.cache[chunk.chunk_hash] = {}
            self.cache[chunk.chunk_hash][chunk.sequence_number] = chunk

    def cleanup_expired(self, ts: datetime):
        """Remove expired chunks"""
        for h in list(self.cache.keys()):
            for s in list(self.cache[h].keys()):
                if self.cache[h][s].is_expired(ts, self.ttl):
                    del self.cache[h][s]
            if not self.cache[h]:
                del self.cache[h]

    def analyze(self, index: int, req_id: str, ts: datetime, body_str: str,
                response_str: Optional[str], req_type: str, msg_count: int):
        """Analyze one request"""

        body = json.loads(body_str)

        # Extract API metrics if available
        api_cache_read = None
        api_cache_creation = None
        api_input_tokens = None

        if response_str:
            try:
                resp = json.loads(response_str)
                usage = resp.get('body', {}).get('usage') if 'body' in resp else resp.get('usage', {})
                if usage:
                    api_cache_read = usage.get('cache_read_input_tokens')
                    api_cache_creation = usage.get('cache_creation_input_tokens')
                    api_input_tokens = usage.get('input_tokens')
            except:
                pass

        # Run simulation
        all_tokens = self.extract_all_tokens(body)
        self.cleanup_expired(ts)
        chunks = self.chunk_tokens(all_tokens, ts)

        cache_read_tokens, matched_chunks = self.find_prefix_match(chunks, ts)
        cache_creation_tokens = len(all_tokens) - cache_read_tokens

        self.update_cache(chunks, matched_chunks)

        cache_hit_rate = cache_read_tokens / len(all_tokens) if all_tokens else 0.0

        analysis = RequestAnalysis(
            index=index,
            request_id=req_id[:12],
            request_type=req_type,
            message_count=msg_count,
            timestamp=ts,
            total_tokens=len(all_tokens),
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_hit_rate=cache_hit_rate,
            api_cache_read=api_cache_read,
            api_cache_creation=api_cache_creation,
            api_input_tokens=api_input_tokens
        )

        self.analyses.append(analysis)
        return analysis

def classify_request(body_str: str) -> str:
    """Classify request type based on stream flag."""
    body = json.loads(body_str)
    stream = body.get('stream')

    if stream is True:
        return 'streaming'
    elif stream is False or stream is None:
        return 'non_streaming'
    return 'unknown'

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Complete cache analysis')
    parser.add_argument('db', help='Path to requests.db')
    parser.add_argument('--conversation-id', required=True)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--cache-ttl', type=int, default=300)
    parser.add_argument('--limit', type=int, help='Limit to first N requests')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    c = conn.cursor()

    # Get indexed requests for time range
    indexed_reqs = c.execute("""
        SELECT r.id, r.timestamp
        FROM requests r
        INNER JOIN conversation_index ci ON r.id = ci.request_id
        WHERE ci.conversation_id = ?
        ORDER BY r.timestamp
    """, (args.conversation_id,)).fetchall()

    if not indexed_reqs:
        print(f"No requests found")
        return

    indexed_ids = {r[0] for r in indexed_reqs}

    # Get time range with buffer
    first_ts = datetime.fromisoformat(indexed_reqs[0][1].replace('Z', '+00:00'))
    last_ts = datetime.fromisoformat(indexed_reqs[-1][1].replace('Z', '+00:00'))
    range_start = (first_ts - timedelta(seconds=300)).isoformat()
    range_end = (last_ts + timedelta(seconds=300)).isoformat()

    # Get ALL requests
    query = """
        SELECT id, timestamp, body, response
        FROM requests
        WHERE response IS NOT NULL
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """

    if args.limit:
        query += f" LIMIT {args.limit * 2}"  # Account for pairs

    all_reqs = c.execute(query, (range_start, range_end)).fetchall()

    print(f"Conversation: {args.conversation_id}")
    print(f"Indexed requests: {len(indexed_ids)}")
    print(f"Total in time range: {len(all_reqs)}")
    print()

    # Filter to conversation using proximity
    def extract_hashes(body_str: str) -> Tuple[str, str, str, int]:
        """Extract content hashes"""
        body = json.loads(body_str)
        messages = body.get('messages', [])
        tools = body.get('tools', [])
        system = body.get('system', [])

        msg_hash = hashlib.md5(json.dumps(messages, separators=(',', ':')).encode()).hexdigest()[:16]
        tool_hash = hashlib.md5(json.dumps(tools, separators=(',', ':')).encode()).hexdigest()[:16]
        sys_hash = hashlib.md5(json.dumps(system, separators=(',', ':')).encode()).hexdigest()[:16]

        return msg_hash, tool_hash, sys_hash, len(messages)

    indexed_info = []
    for req_id, ts_str, body_str, resp_str in all_reqs:
        if req_id in indexed_ids:
            msg_hash, tool_hash, sys_hash, msg_count = extract_hashes(body_str)
            indexed_info.append((req_id, ts_str, tool_hash, sys_hash))

    # Include requests close to indexed with matching hashes
    conversation_reqs = []

    for req_id, ts_str, body_str, resp_str in all_reqs:
        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

        if req_id in indexed_ids:
            conversation_reqs.append((req_id, ts_str, body_str, resp_str))
            continue

        # Check proximity to indexed
        msg_hash, tool_hash, sys_hash, msg_count = extract_hashes(body_str)

        for idx_id, idx_ts, idx_tool_hash, idx_sys_hash in indexed_info:
            idx_dt = datetime.fromisoformat(idx_ts.replace('Z', '+00:00'))
            gap = abs((ts - idx_dt).total_seconds())

            if gap < 30 and tool_hash == idx_tool_hash and sys_hash == idx_sys_hash:
                conversation_reqs.append((req_id, ts_str, body_str, resp_str))
                break

    print(f"Filtered to conversation: {len(conversation_reqs)} requests")
    print()

    # Run simulation
    print("Running cache simulation...")
    sim = CompleteCacheSimulator(args.block_size, args.cache_ttl)

    for i, (req_id, ts_str, body_str, resp_str) in enumerate(conversation_reqs):
        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        body = json.loads(body_str)
        req_type = classify_request(body_str)
        msg_count = len(body.get('messages', []))

        sim.analyze(i, req_id, ts, body_str, resp_str, req_type, msg_count)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(conversation_reqs)}", end='\r')

    print(f"  Processed {len(conversation_reqs)}/{len(conversation_reqs)}")
    print()

    # Calculate statistics
    print("="*80)
    print("COMPLETE CACHE STATISTICS")
    print("="*80)

    total_tokens = sum(a.total_tokens for a in sim.analyses)
    cache_read = sum(a.cache_read_tokens for a in sim.analyses)
    cache_creation = sum(a.cache_creation_tokens for a in sim.analyses)

    print(f"\nALL REQUESTS (streaming + non-streaming):")
    print(f"  Total requests: {len(sim.analyses)}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Cache read: {cache_read:,} ({cache_read/total_tokens*100:.1f}%)")
    print(f"  Cache creation: {cache_creation:,} ({cache_creation/total_tokens*100:.1f}%)")

    # Break down by type
    streaming_analyses = [a for a in sim.analyses if a.request_type == 'streaming']
    non_streaming_analyses = [a for a in sim.analyses if a.request_type == 'non_streaming']

    if streaming_analyses:
        s_total = sum(a.total_tokens for a in streaming_analyses)
        s_read = sum(a.cache_read_tokens for a in streaming_analyses)
        print(f"\nSTREAMING ONLY:")
        print(f"  Requests: {len(streaming_analyses)}")
        print(f"  Total tokens: {s_total:,}")
        print(f"  Cache read: {s_read:,} ({s_read/s_total*100:.1f}%)")

    if non_streaming_analyses:
        ns_total = sum(a.total_tokens for a in non_streaming_analyses)
        ns_read = sum(a.cache_read_tokens for a in non_streaming_analyses)
        print(f"\nNON-STREAMING ONLY (indexed in conversation_index):")
        print(f"  Requests: {len(non_streaming_analyses)}")
        print(f"  Total tokens: {ns_total:,}")
        print(f"  Cache read: {ns_read:,} ({ns_read/ns_total*100:.1f}%)")

    # API Validation - ALL requests with usage data
    all_with_usage = [a for a in sim.analyses
                      if a.api_cache_read is not None or a.api_cache_creation is not None or a.api_input_tokens is not None]

    streaming_with_usage = [a for a in streaming_analyses
                           if a.api_cache_read is not None or a.api_cache_creation is not None or a.api_input_tokens is not None]

    non_streaming_with_usage = [a for a in non_streaming_analyses
                               if a.api_cache_read is not None or a.api_cache_creation is not None or a.api_input_tokens is not None]

    if all_with_usage:
        print(f"\n{'='*80}")
        print("API VALIDATION - ALL REQUESTS")
        print(f"{'='*80}")

        # Calculate totals for all requests with usage
        sim_total = sum(a.total_tokens for a in all_with_usage)
        sim_read = sum(a.cache_read_tokens for a in all_with_usage)
        sim_create = sum(a.cache_creation_tokens for a in all_with_usage)
        sim_input = sim_total - sim_read - sim_create

        api_read = sum(a.api_cache_read or 0 for a in all_with_usage)
        api_create = sum(a.api_cache_creation or 0 for a in all_with_usage)
        api_input = sum(a.api_input_tokens or 0 for a in all_with_usage)
        api_total = api_read + api_create + api_input

        print(f"\nRequests with usage data: {len(all_with_usage)}")
        print(f"  Streaming: {len(streaming_with_usage)}")
        print(f"  Non-streaming: {len(non_streaming_with_usage)}")

        print(f"\nINPUT TOKENS:")
        print(f"  Simulated Total:    {sim_total:,}")
        print(f"    Cache Read:       {sim_read:,} ({sim_read/sim_total*100:.1f}%)")
        print(f"    Cache Creation:   {sim_create:,} ({sim_create/sim_total*100:.1f}%)")
        print(f"    Input (new):      {sim_input:,} ({sim_input/sim_total*100:.1f}%)")
        print()
        print(f"  API Actual Total:   {api_total:,}")
        print(f"    Cache Read:       {api_read:,} ({api_read/api_total*100:.1f}%)")
        print(f"    Cache Creation:   {api_create:,} ({api_create/api_total*100:.1f}%)")
        print(f"    Input (new):      {api_input:,} ({api_input/api_total*100:.1f}%)")
        print()
        print(f"  Comparison:")
        print(f"    Total Accuracy:   {sim_total/api_total*100:.1f}%")
        if api_read > 0:
            print(f"    Cache Read:       {sim_read/sim_total*100:.1f}% vs {api_read/api_total*100:.1f}% (accuracy: {sim_read/api_read*100:.1f}%)")
        if api_create > 0:
            print(f"    Cache Creation:   {sim_create/sim_total*100:.1f}% vs {api_create/api_total*100:.1f}% (accuracy: {sim_create/api_create*100:.1f}%)")

        # Breakdown by request type
        if streaming_with_usage:
            print(f"\n{'-'*60}")
            print("STREAMING REQUESTS ONLY:")
            print(f"{'-'*60}")

            s_sim_total = sum(a.total_tokens for a in streaming_with_usage)
            s_sim_read = sum(a.cache_read_tokens for a in streaming_with_usage)
            s_api_read = sum(a.api_cache_read or 0 for a in streaming_with_usage)
            s_api_total = sum((a.api_cache_read or 0) + (a.api_cache_creation or 0) + (a.api_input_tokens or 0) for a in streaming_with_usage)

            print(f"  Requests: {len(streaming_with_usage)}")
            print(f"  Simulated cache read: {s_sim_read:,} / {s_sim_total:,} ({s_sim_read/s_sim_total*100:.1f}%)")
            print(f"  API cache read:       {s_api_read:,} / {s_api_total:,} ({s_api_read/s_api_total*100:.1f}%)")
            if s_api_read > 0:
                print(f"  Accuracy:             {s_sim_read/s_api_read*100:.1f}%")

        if non_streaming_with_usage:
            print(f"\n{'-'*60}")
            print("NON-STREAMING REQUESTS ONLY:")
            print(f"{'-'*60}")

            ns_sim_total = sum(a.total_tokens for a in non_streaming_with_usage)
            ns_sim_read = sum(a.cache_read_tokens for a in non_streaming_with_usage)
            ns_api_read = sum(a.api_cache_read or 0 for a in non_streaming_with_usage)
            ns_api_total = sum((a.api_cache_read or 0) + (a.api_cache_creation or 0) + (a.api_input_tokens or 0) for a in non_streaming_with_usage)

            print(f"  Requests: {len(non_streaming_with_usage)}")
            print(f"  Simulated cache read: {ns_sim_read:,} / {ns_sim_total:,} ({ns_sim_read/ns_sim_total*100:.1f}%)")
            print(f"  API cache read:       {ns_api_read:,} / {ns_api_total:,} ({ns_api_read/ns_api_total*100:.1f}%)")
            if ns_api_read > 0:
                print(f"  Accuracy:             {ns_sim_read/ns_api_read*100:.1f}%")

    conn.close()

if __name__ == "__main__":
    main()
