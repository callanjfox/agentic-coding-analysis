#!/usr/bin/env python3
"""
Recover Conversations from requests.db Without JSONL Files

Reconstructs conversation chains by:
1. Pairing streaming/non-streaming requests (same content, ~5-8s apart)
2. Chaining pairs by hash prefix overlap (97%+ within conversation)
3. Using message count progression (1→3→5→7) for validation

Output: Mock JSONL files in recovered_jsonl/ directory that can be used
by build_minimal_traces.py for trace generation.

Usage:
    # Recover all conversations
    python3 recover_conversations.py request_dbs/requests_from_locallinux.db \
        --output-dir recovered_jsonl/

    # Validate against existing JSONL
    python3 recover_conversations.py request_dbs/requests_from_locallinux.db \
        --output-dir recovered_jsonl/ \
        --validate-against jsonl/

    # Exclude already-indexed requests
    python3 recover_conversations.py request_dbs/requests_from_locallinux.db \
        --output-dir recovered_jsonl/ \
        --exclude-indexed jsonl/
"""

import sqlite3
import json
import hashlib
import tiktoken
import argparse
import uuid
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ProcessedRequest:
    """A preprocessed request with computed hashes"""
    request_id: str
    timestamp: datetime
    req_type: str  # 'streaming' or 'non_streaming'
    msg_count: int  # len(messages)
    tool_hash: str  # MD5[:16] of tools section
    sys_hash: str   # MD5[:16] of system section
    content_hash: str  # MD5 of normalized full content
    hash_ids: List[int] = field(default_factory=list)  # Block hash IDs
    message_id: str = ""  # From response, for validation/output
    body_str: str = ""  # Raw body for later processing


@dataclass
class RequestPair:
    """A streaming + non-streaming pair forming one turn"""
    streaming: ProcessedRequest
    non_streaming: ProcessedRequest
    turn_number: int = 0
    conversation_id: str = ""


@dataclass
class ConversationChain:
    """A recovered conversation"""
    conversation_id: str
    pairs: List[RequestPair] = field(default_factory=list)


# Standalone function for multiprocessing (can't pickle methods)
def compute_hash_ids_standalone(body_str: str, block_size: int = 64) -> List[int]:
    """Compute block hash IDs for a request body (standalone for multiprocessing)"""
    import copy

    tokenizer = tiktoken.encoding_for_model("gpt-4")

    def normalize_for_cache(obj):
        normalized = copy.deepcopy(obj)
        def remove_markers(item):
            if isinstance(item, dict):
                item.pop('cache_control', None)
                item.pop('signature', None)
                for value in item.values():
                    remove_markers(value)
            elif isinstance(item, list):
                for element in item:
                    remove_markers(element)
        remove_markers(normalized)
        return normalized

    def create_chained_hash(tokens, prev_hash, seq_num):
        content = f"{prev_hash}:{seq_num}:" + " ".join(map(str, tokens))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    body = json.loads(body_str)
    tools = body.get('tools', [])
    system = body.get('system', [])
    messages = body.get('messages', [])

    tools_norm = normalize_for_cache(tools)
    system_norm = normalize_for_cache(system)
    messages_norm = normalize_for_cache(messages)

    all_tokens = []
    if tools:
        all_tokens.extend(tokenizer.encode(json.dumps(tools_norm, separators=(',', ':'))))
    if system:
        all_tokens.extend(tokenizer.encode(json.dumps(system_norm, separators=(',', ':'))))
    if messages:
        all_tokens.extend(tokenizer.encode(json.dumps(messages_norm, separators=(',', ':'))))

    # Create chained block hashes - ONLY FULL BLOCKS
    hash_ids = []
    prev_hash = "0"
    hash_to_id = {}
    next_id = 1

    for seq_num in range(len(all_tokens) // block_size):
        i = seq_num * block_size
        block_tokens = all_tokens[i:i + block_size]
        chunk_hash = create_chained_hash(block_tokens, prev_hash, seq_num)

        if chunk_hash not in hash_to_id:
            hash_to_id[chunk_hash] = next_id
            next_id += 1

        hash_ids.append(hash_to_id[chunk_hash])
        prev_hash = chunk_hash

    return hash_ids


class ConversationRecoverer:
    """Recovers conversations from database without JSONL files"""

    def __init__(self, block_size: int = 64):
        self.block_size = block_size
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Hash management
        self.hash_to_id: Dict[str, int] = {}
        self.next_hash_id = 1

    def get_hash_id(self, chunk_hash: str) -> int:
        """Get or create numeric ID for a hash"""
        if chunk_hash not in self.hash_to_id:
            self.hash_to_id[chunk_hash] = self.next_hash_id
            self.next_hash_id += 1
        return self.hash_to_id[chunk_hash]

    def create_chained_hash(self, token_ids: List[int], prev_hash: str, seq_num: int) -> str:
        """Create a hash that chains with previous block"""
        content = f"{prev_hash}:{seq_num}:" + " ".join(map(str, token_ids))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def normalize_for_cache(self, obj) -> any:
        """Remove cache_control and signature markers recursively"""
        import copy
        normalized = copy.deepcopy(obj)

        def remove_markers(item):
            if isinstance(item, dict):
                item.pop('cache_control', None)
                item.pop('signature', None)
                for value in item.values():
                    remove_markers(value)
            elif isinstance(item, list):
                for element in item:
                    remove_markers(element)

        remove_markers(normalized)
        return normalized

    def classify_request(self, body_str: str) -> str:
        """Classify request type based on stream flag."""
        body = json.loads(body_str)
        stream = body.get('stream')

        if stream is True:
            return 'streaming'
        elif stream is False or stream is None:
            return 'non_streaming'
        return 'unknown'

    def extract_hashes(self, body_str: str) -> Tuple[str, str, str]:
        """Extract tool_hash, sys_hash, and content_hash from request body

        content_hash is based on tools+system+messages only (not max_tokens, stream, etc.)
        This allows matching streaming and non-streaming pairs.
        """
        body = json.loads(body_str)

        tools = body.get('tools', [])
        system = body.get('system', [])
        messages = body.get('messages', [])

        # Normalize before hashing
        tools_norm = self.normalize_for_cache(tools)
        system_norm = self.normalize_for_cache(system)
        messages_norm = self.normalize_for_cache(messages)

        tool_hash = hashlib.md5(json.dumps(tools_norm, separators=(',', ':')).encode()).hexdigest()[:16]
        sys_hash = hashlib.md5(json.dumps(system_norm, separators=(',', ':')).encode()).hexdigest()[:16]

        # Content hash: only tools + system + messages (what matters for caching)
        content_for_hash = {
            'tools': tools_norm,
            'system': system_norm,
            'messages': messages_norm
        }
        content_hash = hashlib.md5(json.dumps(content_for_hash, separators=(',', ':')).encode()).hexdigest()

        return tool_hash, sys_hash, content_hash

    def extract_message_id(self, response_str: Optional[str]) -> str:
        """Extract message ID from response"""
        if not response_str:
            return ""
        try:
            resp = json.loads(response_str)
            body = resp.get('body', resp)
            return body.get('id', '')
        except:
            return ""

    def compute_hash_ids(self, body_str: str) -> List[int]:
        """Compute block hash IDs for a request"""
        body = json.loads(body_str)

        tools = body.get('tools', [])
        system = body.get('system', [])
        messages = body.get('messages', [])

        # Normalize and tokenize
        tools_norm = self.normalize_for_cache(tools)
        system_norm = self.normalize_for_cache(system)
        messages_norm = self.normalize_for_cache(messages)

        tools_tokens = self.tokenizer.encode(json.dumps(tools_norm, separators=(',', ':'))) if tools else []
        system_tokens = self.tokenizer.encode(json.dumps(system_norm, separators=(',', ':'))) if system else []
        messages_tokens = self.tokenizer.encode(json.dumps(messages_norm, separators=(',', ':'))) if messages else []

        all_tokens = tools_tokens + system_tokens + messages_tokens

        # Create chained block hashes - ONLY FULL BLOCKS
        hash_ids = []
        prev_hash = "0"
        full_block_count = len(all_tokens) // self.block_size

        for seq_num in range(full_block_count):
            i = seq_num * self.block_size
            block_tokens = all_tokens[i:i + self.block_size]
            chunk_hash = self.create_chained_hash(block_tokens, prev_hash, seq_num)
            hash_id = self.get_hash_id(chunk_hash)
            hash_ids.append(hash_id)
            prev_hash = chunk_hash

        return hash_ids

    def count_content_tokens(self, content) -> int:
        """Count tokens in tools or system content"""
        if not content:
            return 0
        return len(self.tokenizer.encode(json.dumps(content, separators=(',', ':'))))

    def load_requests(self, db_path: Path, exclude_msg_ids: Optional[Set[str]] = None,
                       limit: Optional[int] = None,
                       min_tool_system_tokens: int = 5000) -> List[ProcessedRequest]:
        """Load and preprocess all requests from database

        Args:
            min_tool_system_tokens: Minimum combined tool + system tokens to include.
                                   Claude Code has ~15K (12K tools + 3K system).
                                   Non-Claude-Code typically has ~60 tokens or less.
                                   Default 5000 filters out non-Claude-Code requests.
        """
        import sys
        print(f"Loading requests from {db_path}...", flush=True)
        print(f"  Filtering: min_tool_system_tokens >= {min_tool_system_tokens}", flush=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all requests with Claude Code patterns
        query = """
            SELECT id, timestamp, body, response
            FROM requests
            WHERE body LIKE '%"max_tokens"%'
            ORDER BY timestamp
        """
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)

        requests = []
        skipped = 0
        skipped_non_claude_code = 0

        for row in cursor.fetchall():
            req_id, ts_str, body_str, response_str = row

            try:
                req_type = self.classify_request(body_str)
                if req_type == 'unknown':
                    continue

                body = json.loads(body_str)

                # Count tool and system tokens
                tools = body.get('tools', [])
                system = body.get('system', [])
                tool_tokens = self.count_content_tokens(tools)
                system_tokens = self.count_content_tokens(system)

                # Filter out non-Claude-Code requests
                # Claude Code has ~12K tool tokens + ~3K system tokens
                # Web interface/API has ~0-60 tokens
                if tool_tokens + system_tokens < min_tool_system_tokens:
                    skipped_non_claude_code += 1
                    continue

                msg_count = len(body.get('messages', []))

                tool_hash, sys_hash, content_hash = self.extract_hashes(body_str)
                message_id = self.extract_message_id(response_str)

                # Skip if already indexed
                if exclude_msg_ids and message_id in exclude_msg_ids:
                    skipped += 1
                    continue

                # Parse timestamp
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

                req = ProcessedRequest(
                    request_id=req_id,
                    timestamp=ts,
                    req_type=req_type,
                    msg_count=msg_count,
                    tool_hash=tool_hash,
                    sys_hash=sys_hash,
                    content_hash=content_hash,
                    message_id=message_id,
                    body_str=body_str
                )
                requests.append(req)

            except Exception as e:
                continue

        conn.close()

        print(f"  Loaded {len(requests)} Claude Code requests")
        print(f"    Skipped: {skipped_non_claude_code} non-Claude-Code (tool+sys tokens < {min_tool_system_tokens})")
        print(f"    Skipped: {skipped} already indexed")
        print(f"  Streaming: {sum(1 for r in requests if r.req_type == 'streaming')}")
        print(f"  Non-streaming: {sum(1 for r in requests if r.req_type == 'non_streaming')}")

        return requests

    def pair_requests(self, requests: List[ProcessedRequest]) -> Tuple[List[RequestPair], List[ProcessedRequest]]:
        """Pair streaming + non-streaming requests that form a single turn"""
        print("Pairing streaming + non-streaming requests...")

        paired = []
        used = set()

        # Sort by timestamp
        sorted_reqs = sorted(requests, key=lambda r: r.timestamp)

        # Build content_hash index for faster lookup
        streaming_by_hash = defaultdict(list)
        for req in sorted_reqs:
            if req.req_type == 'streaming':
                streaming_by_hash[req.content_hash].append(req)

        for req in sorted_reqs:
            if req.request_id in used or req.req_type != 'non_streaming':
                continue

            # Find streaming pair with same content hash
            candidates = streaming_by_hash.get(req.content_hash, [])

            best_pair = None
            best_gap = float('inf')

            for stream_req in candidates:
                if stream_req.request_id in used:
                    continue

                gap = (req.timestamp - stream_req.timestamp).total_seconds()

                # Streaming should come before non-streaming, within 30 seconds
                if 0 < gap <= 30 and gap < best_gap:
                    best_pair = stream_req
                    best_gap = gap

            if best_pair:
                pair = RequestPair(streaming=best_pair, non_streaming=req)
                paired.append(pair)
                used.add(best_pair.request_id)
                used.add(req.request_id)

        orphans = [r for r in sorted_reqs if r.request_id not in used]

        print(f"  Created {len(paired)} pairs")
        print(f"  Orphans: {len(orphans)}")

        return paired, orphans

    def compute_pair_hashes(self, pairs: List[RequestPair], workers: int = 4):
        """Compute hash_ids for each pair using multiprocessing"""
        import sys
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"Computing block hashes for {len(pairs)} pairs ({workers} workers)...", flush=True)

        # Prepare work items
        body_strs = [pair.streaming.body_str for pair in pairs]

        # Use multiprocessing
        results = [None] * len(pairs)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all jobs
            future_to_idx = {}
            for i, body_str in enumerate(body_strs):
                future = executor.submit(compute_hash_ids_standalone, body_str, self.block_size)
                future_to_idx[future] = i

            # Collect results
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                completed += 1
                if completed % 500 == 0:
                    print(f"  Completed {completed}/{len(pairs)}...", flush=True)

        # Assign results to pairs
        for i, pair in enumerate(pairs):
            pair.streaming.hash_ids = results[i]
            pair.non_streaming.hash_ids = results[i]

        print(f"  Done computing hashes", flush=True)

    def prefix_overlap(self, prev_hash_ids: List[int], curr_hash_ids: List[int]) -> float:
        """Calculate fraction of previous hash_ids preserved in current"""
        if not prev_hash_ids:
            return 0.0

        # Count matching prefix
        prefix_len = 0
        for a, b in zip(prev_hash_ids, curr_hash_ids):
            if a == b:
                prefix_len += 1
            else:
                break

        return prefix_len / len(prev_hash_ids)

    def chain_conversations(self, pairs: List[RequestPair]) -> List[ConversationChain]:
        """Chain pairs into conversations using hash prefix overlap.

        Optimized: Use first hash_id as index to avoid O(n²) comparisons.
        If 90%+ prefix overlap exists, first hash_id must match.
        """
        import sys
        print("Chaining pairs into conversations...", flush=True)

        # Sort by timestamp
        sorted_pairs = sorted(pairs, key=lambda p: p.streaming.timestamp)

        conversations: List[ConversationChain] = []
        pair_to_conv: Dict[str, ConversationChain] = {}  # pair_id -> conversation
        assigned = set()

        # Build index by first hash_id for fast lookup
        # pairs_by_first_hash[first_hash] = [(pair, idx)]
        pairs_by_first_hash: Dict[int, List[Tuple[RequestPair, int]]] = defaultdict(list)

        for idx, pair in enumerate(sorted_pairs):
            if pair.streaming.hash_ids:
                first_hash = pair.streaming.hash_ids[0]
                pairs_by_first_hash[first_hash].append((pair, idx))

        # Process pairs in timestamp order with decreasing thresholds
        for threshold in [0.90, 0.80, 0.70, 0.50]:
            newly_assigned = 0

            for pair_idx, pair in enumerate(sorted_pairs):
                if pair.streaming.request_id in assigned:
                    continue

                curr_hash_ids = pair.streaming.hash_ids
                curr_msg_count = pair.streaming.msg_count

                if not curr_hash_ids:
                    continue

                # Find best predecessor - only check pairs with matching first hash
                best_pred = None
                best_overlap = 0
                best_conv = None

                first_hash = curr_hash_ids[0]
                candidates = pairs_by_first_hash.get(first_hash, [])

                for prev_pair, prev_idx in candidates:
                    # Must be assigned and before current pair
                    if prev_pair.streaming.request_id not in assigned:
                        continue
                    if prev_idx >= pair_idx:
                        continue

                    # Current should have MORE blocks than predecessor
                    if len(curr_hash_ids) <= len(prev_pair.streaming.hash_ids):
                        continue

                    # Message count should INCREASE (1→3→5→7...)
                    # If it drops or stays same, this is a NEW conversation
                    prev_msg_count = prev_pair.streaming.msg_count
                    if curr_msg_count <= prev_msg_count:
                        continue

                    # Time gap check - max 1 hour between turns
                    # (Most conversation turns are < 10 minutes apart)
                    gap = (pair.streaming.timestamp - prev_pair.streaming.timestamp).total_seconds()
                    if gap > 3600:  # 1 hour max
                        continue

                    # Calculate hash prefix overlap
                    overlap = self.prefix_overlap(prev_pair.streaming.hash_ids, curr_hash_ids)

                    if overlap >= threshold and overlap > best_overlap:
                        best_pred = prev_pair
                        best_overlap = overlap
                        best_conv = pair_to_conv.get(prev_pair.streaming.request_id)

                if best_pred and best_conv:
                    # Continue existing conversation
                    pair.turn_number = best_pred.turn_number + 1
                    pair.conversation_id = best_conv.conversation_id
                    best_conv.pairs.append(pair)
                    pair_to_conv[pair.streaming.request_id] = best_conv
                    assigned.add(pair.streaming.request_id)
                    newly_assigned += 1
                elif threshold == 0.90:
                    # First pass: also start new conversations
                    conv_id = f"recovered-{uuid.uuid4().hex[:12]}"
                    conv = ConversationChain(conversation_id=conv_id)
                    pair.turn_number = (curr_msg_count + 1) // 2  # Estimate from msg_count
                    pair.conversation_id = conv_id
                    conv.pairs.append(pair)
                    conversations.append(conv)
                    pair_to_conv[pair.streaming.request_id] = conv
                    assigned.add(pair.streaming.request_id)
                    newly_assigned += 1

            unassigned_count = sum(1 for p in sorted_pairs if p.streaming.request_id not in assigned)
            print(f"  Pass with threshold {threshold}: assigned {newly_assigned}, remaining {unassigned_count}", flush=True)

        # Handle remaining unassigned
        unassigned_count = sum(1 for p in sorted_pairs if p.streaming.request_id not in assigned)
        if unassigned_count > 0:
            print(f"  {unassigned_count} pairs could not be chained (creating single-pair conversations)")
            for pair in sorted_pairs:
                if pair.streaming.request_id not in assigned:
                    conv_id = f"recovered-orphan-{uuid.uuid4().hex[:8]}"
                    conv = ConversationChain(conversation_id=conv_id)
                    pair.turn_number = (pair.streaming.msg_count + 1) // 2
                    pair.conversation_id = conv_id
                    conv.pairs.append(pair)
                    conversations.append(conv)
                    assigned.add(pair.streaming.request_id)

        print(f"  Recovered {len(conversations)} conversations")

        # Sort pairs within each conversation
        for conv in conversations:
            conv.pairs.sort(key=lambda p: p.streaming.timestamp)

        return conversations

    def generate_mock_jsonl(self, conversations: List[ConversationChain], output_dir: Path):
        """Generate mock JSONL files for each recovered conversation"""
        print(f"Generating mock JSONL files to {output_dir}/...")
        output_dir.mkdir(parents=True, exist_ok=True)

        for conv in conversations:
            if not conv.pairs:
                continue

            jsonl_path = output_dir / f"{conv.conversation_id}.jsonl"

            with open(jsonl_path, 'w') as f:
                for pair in conv.pairs:
                    # Use non-streaming message ID (that's what real JSONL has)
                    msg_id = pair.non_streaming.message_id
                    if not msg_id:
                        msg_id = pair.streaming.message_id

                    if msg_id:
                        entry = {
                            "type": "assistant",
                            "message": {"id": msg_id},
                            "timestamp": pair.non_streaming.timestamp.isoformat(),
                            "_recovered": True,  # Mark as recovered
                            "_turn": pair.turn_number,
                            "_msg_count": pair.non_streaming.msg_count
                        }
                        f.write(json.dumps(entry) + '\n')

        print(f"  Generated {len(conversations)} mock JSONL files")

    def validate_against_jsonl(self, conversations: List[ConversationChain],
                                jsonl_dir: Path, db_path: Path) -> Dict:
        """Validate recovered conversations against existing JSONL ground truth"""
        print(f"Validating against {jsonl_dir}...")

        # Build ground truth: message_id -> conversation_id
        gt_msg_to_conv: Dict[str, str] = {}
        gt_conv_requests: Dict[str, List[str]] = defaultdict(list)

        for jsonl_file in jsonl_dir.glob("*.jsonl"):
            if jsonl_file.name.startswith("agent-"):
                continue  # Skip sub-agents
            if jsonl_file.name.startswith("recovered"):
                continue  # Skip our own output

            conv_id = jsonl_file.stem

            with open(jsonl_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('type') == 'assistant':
                            msg = entry.get('message', {})
                            msg_id = msg.get('id', '')
                            if msg_id:
                                gt_msg_to_conv[msg_id] = conv_id
                                gt_conv_requests[conv_id].append(msg_id)
                    except:
                        continue

        print(f"  Ground truth: {len(gt_conv_requests)} conversations, {len(gt_msg_to_conv)} message IDs")

        # Check our recovered conversations
        correct = 0
        fragmented = 0
        total_recovered = 0

        for conv in conversations:
            if not conv.pairs:
                continue

            # Get message IDs in this recovered conversation
            msg_ids = []
            for pair in conv.pairs:
                if pair.non_streaming.message_id:
                    msg_ids.append(pair.non_streaming.message_id)

            if not msg_ids:
                continue

            total_recovered += 1

            # Check which ground truth conversations these belong to
            gt_convs = set()
            for msg_id in msg_ids:
                if msg_id in gt_msg_to_conv:
                    gt_convs.add(gt_msg_to_conv[msg_id])

            if len(gt_convs) == 1:
                correct += 1
            elif len(gt_convs) > 1:
                fragmented += 1
                # This recovered conversation spans multiple GT conversations

        accuracy = correct / total_recovered if total_recovered > 0 else 0

        result = {
            'total_recovered': total_recovered,
            'correct': correct,
            'fragmented': fragmented,
            'accuracy': accuracy,
            'gt_conversations': len(gt_conv_requests),
            'gt_message_ids': len(gt_msg_to_conv)
        }

        print(f"  Accuracy: {accuracy:.1%} ({correct}/{total_recovered})")
        print(f"  Fragmented: {fragmented}")

        return result


def get_indexed_message_ids(jsonl_dir: Path) -> Set[str]:
    """Get all message IDs from existing JSONL files"""
    msg_ids = set()

    for jsonl_file in jsonl_dir.glob("*.jsonl"):
        if jsonl_file.name.startswith("recovered"):
            continue

        with open(jsonl_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry.get('type') == 'assistant':
                        msg = entry.get('message', {})
                        msg_id = msg.get('id', '')
                        if msg_id:
                            msg_ids.add(msg_id)
                except:
                    continue

    return msg_ids


def main():
    parser = argparse.ArgumentParser(description='Recover conversations from requests.db')
    parser.add_argument('db_path', type=Path, help='Path to requests database')
    parser.add_argument('--output-dir', type=Path, default=Path('recovered_jsonl'),
                        help='Output directory for mock JSONL files')
    parser.add_argument('--validate-against', type=Path,
                        help='JSONL directory to validate against')
    parser.add_argument('--exclude-indexed', type=Path,
                        help='JSONL directory - exclude already-indexed requests')
    parser.add_argument('--block-size', type=int, default=64,
                        help='Block size for hash computation')
    parser.add_argument('--limit', type=int,
                        help='Limit number of requests to process (for testing)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers for hash computation')

    args = parser.parse_args()

    recoverer = ConversationRecoverer(block_size=args.block_size)

    # Get excluded message IDs if specified
    exclude_ids = None
    if args.exclude_indexed:
        exclude_ids = get_indexed_message_ids(args.exclude_indexed)
        print(f"Excluding {len(exclude_ids)} already-indexed message IDs")

    # Load and process
    requests = recoverer.load_requests(args.db_path, exclude_ids, limit=args.limit)

    if not requests:
        print("No requests to process")
        return

    pairs, orphans = recoverer.pair_requests(requests)

    if not pairs:
        print("No pairs created")
        return

    recoverer.compute_pair_hashes(pairs, workers=args.workers)
    conversations = recoverer.chain_conversations(pairs)

    # Generate output
    recoverer.generate_mock_jsonl(conversations, args.output_dir)

    # Validate if requested
    if args.validate_against:
        recoverer.validate_against_jsonl(conversations, args.validate_against, args.db_path)

    # Summary
    print("\n=== Summary ===")
    print(f"Total requests: {len(requests)}")
    print(f"Paired: {len(pairs) * 2}")
    print(f"Orphans: {len(orphans)}")
    print(f"Conversations recovered: {len(conversations)}")

    # Stats on conversation sizes
    sizes = [len(c.pairs) for c in conversations]
    print(f"Conversation sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")


if __name__ == '__main__':
    main()
