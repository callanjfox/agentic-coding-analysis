#!/usr/bin/env python3
"""
Complete Conversation Builder

Builds complete conversation timelines including BOTH streaming and non-streaming requests.
Does not rely solely on conversation_index - uses temporal clustering and content similarity.
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RequestInfo:
    """Complete request information"""
    request_id: str
    timestamp: datetime
    request_type: str  # 'streaming', 'non_streaming', 'unknown'
    max_tokens: int
    stream_flag: bool
    message_count: int
    messages_hash: str
    tools_hash: str
    system_hash: str
    stop_reason: Optional[str]
    in_conversation_index: bool
    paired_with: Optional[str] = None

def classify_request(body_str: str) -> Tuple[str, int, bool]:
    """Classify request as streaming or non-streaming."""
    body = json.loads(body_str)
    max_tokens = body.get('max_tokens', 0)
    stream = body.get('stream')

    if stream is True:
        return 'streaming', max_tokens, True
    elif stream is False or stream is None:
        return 'non_streaming', max_tokens, False
    return 'unknown', max_tokens, bool(stream)

def extract_hashes(body_str: str) -> Tuple[str, str, str, int]:
    """Extract content hashes and message count"""
    body = json.loads(body_str)

    messages = body.get('messages', [])
    tools = body.get('tools', [])
    system = body.get('system', [])

    msg_hash = hashlib.md5(json.dumps(messages, separators=(',', ':')).encode()).hexdigest()[:16]
    tool_hash = hashlib.md5(json.dumps(tools, separators=(',', ':')).encode()).hexdigest()[:16]
    sys_hash = hashlib.md5(json.dumps(system, separators=(',', ':')).encode()).hexdigest()[:16]

    return msg_hash, tool_hash, sys_hash, len(messages)

def get_stop_reason(response_str: Optional[str]) -> Optional[str]:
    """Extract stop_reason from response"""
    if not response_str:
        return None

    try:
        resp = json.loads(response_str)
        if 'body' in resp:
            return resp['body'].get('stop_reason')
        return resp.get('stop_reason')
    except:
        return None

def build_complete_conversation(db_path: str, conversation_id: str, max_gap_minutes: int = 60) -> List[RequestInfo]:
    """
    Build complete conversation timeline including streaming and non-streaming requests.

    Strategy:
    1. Get indexed requests to establish time range
    2. Find ALL requests in that range
    3. Filter to same conversation using content similarity and temporal proximity
    4. Pair streaming/non-streaming requests
    5. Return sorted complete timeline
    """

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Get indexed requests for time range
    indexed_reqs = c.execute("""
        SELECT r.id, r.timestamp
        FROM requests r
        INNER JOIN conversation_index ci ON r.id = ci.request_id
        WHERE ci.conversation_id = ?
        ORDER BY r.timestamp
    """, (conversation_id,)).fetchall()

    if not indexed_reqs:
        print(f"No indexed requests found for {conversation_id}")
        return []

    indexed_ids = {r[0] for r in indexed_reqs}

    # Extend range to catch requests outside indexed bounds
    first_ts = datetime.fromisoformat(indexed_reqs[0][1].replace('Z', '+00:00'))
    last_ts = datetime.fromisoformat(indexed_reqs[-1][1].replace('Z', '+00:00'))

    range_start = (first_ts - timedelta(seconds=300)).isoformat()
    range_end = (last_ts + timedelta(seconds=300)).isoformat()

    # Get ALL requests in range
    all_reqs = c.execute("""
        SELECT id, timestamp, body, response
        FROM requests
        WHERE response IS NOT NULL
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """, (range_start, range_end)).fetchall()

    print(f"Conversation {conversation_id}:")
    print(f"  Indexed requests: {len(indexed_ids)}")
    print(f"  Total in time range: {len(all_reqs)}")

    # Build request info objects
    all_request_info: List[RequestInfo] = []

    for req_id, ts_str, body_str, resp_str in all_reqs:
        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        req_type, max_tokens, stream = classify_request(body_str)
        msg_hash, tool_hash, sys_hash, msg_count = extract_hashes(body_str)
        stop_reason = get_stop_reason(resp_str)
        in_index = req_id in indexed_ids

        info = RequestInfo(
            request_id=req_id,
            timestamp=ts,
            request_type=req_type,
            max_tokens=max_tokens,
            stream_flag=stream,
            message_count=msg_count,
            messages_hash=msg_hash,
            tools_hash=tool_hash,
            system_hash=sys_hash,
            stop_reason=stop_reason,
            in_conversation_index=in_index
        )

        all_request_info.append(info)

    # Filter to likely same-conversation requests
    # Strategy: Include request if:
    # 1. It's indexed, OR
    # 2. It's within 30s of an indexed request AND has matching tool/system hashes

    # First, build indexed_info list for faster lookup
    indexed_info_list = [r for r in all_request_info if r.in_conversation_index]

    conversation_requests = []

    for info in all_request_info:
        if info.in_conversation_index:
            conversation_requests.append(info)
            continue

        # Check if close to any indexed request - OPTIMIZED
        close_to_indexed = False

        # Binary search-like approach: check requests near this timestamp
        for indexed_info in indexed_info_list:
            gap = abs((info.timestamp - indexed_info.timestamp).total_seconds())

            # Skip if too far (requests are sorted)
            if gap > 60:
                continue

            # Within 30s and matching tools/system?
            if gap < 30 and info.tools_hash == indexed_info.tools_hash and info.system_hash == indexed_info.system_hash:
                close_to_indexed = True
                break

        if close_to_indexed:
            conversation_requests.append(info)

    # Pair requests - OPTIMIZED: only check nearby requests
    paired_ids = set()

    for i, info in enumerate(conversation_requests):
        if info.request_id in paired_ids:
            continue

        # Only check next 20 requests (usually within seconds)
        for j in range(i + 1, min(i + 20, len(conversation_requests))):
            other = conversation_requests[j]

            if other.request_id in paired_ids:
                continue

            gap = abs((info.timestamp - other.timestamp).total_seconds())

            # Stop looking if gap > 30s
            if gap > 30:
                break

            if (info.messages_hash == other.messages_hash and
                info.request_type != other.request_type and
                info.request_type in ['streaming', 'non_streaming'] and
                other.request_type in ['streaming', 'non_streaming']):

                # Found pair!
                info.paired_with = other.request_id
                other.paired_with = info.request_id
                paired_ids.add(info.request_id)
                paired_ids.add(other.request_id)
                break

    # Sort by timestamp
    conversation_requests.sort(key=lambda r: r.timestamp)

    # Print summary
    streaming_count = sum(1 for r in conversation_requests if r.request_type == 'streaming')
    non_streaming_count = sum(1 for r in conversation_requests if r.request_type == 'non_streaming')
    paired_count = sum(1 for r in conversation_requests if r.paired_with is not None)

    print(f"\nComplete conversation timeline:")
    print(f"  Total requests: {len(conversation_requests)}")
    print(f"  Streaming: {streaming_count}")
    print(f"  Non-streaming: {non_streaming_count}")
    print(f"  Paired: {paired_count}")
    print(f"  Solo: {len(conversation_requests) - paired_count}")

    conn.close()
    return conversation_requests

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Build complete conversation timeline')
    parser.add_argument('db', help='Path to requests.db')
    parser.add_argument('--conversation-id', required=True)
    parser.add_argument('--show-first', type=int, default=20)
    parser.add_argument('--output', help='Output CSV file')
    args = parser.parse_args()

    requests = build_complete_conversation(args.db, args.conversation_id)

    print(f"\n{'='*100}")
    print(f"FIRST {args.show_first} REQUESTS")
    print(f"{'='*100}")
    print(f"{'Idx':>4} | {'ID':>12} | {'Type':>14} | {'Msgs':>4} | {'Max':>6} | {'Stream':>6} | {'Stop':>12} | {'Paired':>12} | {'Indexed':>7}")
    print("-"*100)

    for i, req in enumerate(requests[:args.show_first]):
        paired_str = req.paired_with[:12] if req.paired_with else '-'
        indexed_str = 'YES' if req.in_conversation_index else 'no'

        print(f"{i:4d} | {req.request_id[:12]:>12} | {req.request_type:>14} | "
              f"{req.message_count:4d} | {req.max_tokens:6d} | {str(req.stream_flag):>6} | "
              f"{req.stop_reason or 'None':>12} | {paired_str:>12} | {indexed_str:>7}")

    # Save to CSV if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write("index,request_id,timestamp,type,messages,max_tokens,stream,stop_reason,paired_with,indexed\n")
            for i, req in enumerate(requests):
                paired = req.paired_with or ''
                indexed = '1' if req.in_conversation_index else '0'
                f.write(f"{i},{req.request_id},{req.timestamp.isoformat()},{req.request_type},"
                       f"{req.message_count},{req.max_tokens},{req.stream_flag},{req.stop_reason or ''},"
                       f"{paired},{indexed}\n")
        print(f"\nSaved to: {args.output}")

if __name__ == "__main__":
    main()
