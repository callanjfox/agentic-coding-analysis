#!/usr/bin/env python3
"""
Build Minimal Traces from Real Conversations

Creates compact conversation traces with full hash_ids for analyzing cache behavior.
Reads directly from JSONL files and requests.db - no pre-built indexes required.

Usage:
    # Process all conversations from JSONL directory
    python3 build_minimal_traces.py requests.db --jsonl-dir jsonl/ --output-dir traces/

    # Process single conversation
    python3 build_minimal_traces.py requests.db --jsonl-dir jsonl/ --conversation-id ff136b44-...

Output format:
{
  "id": "conversation_id",
  "block_size": 256,
  "requests": [
    {
      "t": 0,           // timestamp (seconds from start)
      "type": "s",      // "s" = streaming, "n" = non-streaming
      "in": 18374,      // input tokens
      "out": 250,       // output tokens
      "hash_ids": [...], // full hash_id array for prefix analysis
      "user_msgs": 1,   // user text messages
      "agent_msgs": 0,  // agentic messages (tool_use, tool_result)
      "stop": ""        // stop_reason
    }
  ]
}
"""

import sqlite3
import json
import hashlib
import tiktoken
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any


class MinimalTraceBuilder:
    """Builds minimal traces with full hash_ids.

    IMPORTANT: Only uses FULL blocks (default 256 tokens each).
    Partial chunks at the end are discarded to ensure consistent hashing
    across requests - the last partial chunk would always change as content grows.
    """

    def __init__(self, block_size: int = 256, shared_hash_state: dict = None):
        self.block_size = block_size
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")

        # Hash management - assigns numeric IDs to unique hashes
        # When shared_hash_state is provided, all builders share the same
        # mapping, making hash_ids globally consistent across conversations
        if shared_hash_state is not None:
            self.hash_to_id = shared_hash_state['hash_to_id']
            self._shared_counter = shared_hash_state
        else:
            self.hash_to_id: Dict[str, int] = {}
            self._shared_counter = {'hash_to_id': self.hash_to_id, 'next_id': 1}

        # Results
        self.records: List[Dict] = []
        self.start_ts: Optional[datetime] = None

        # Track previous request for computing client additions
        self.prev_stop_reason: Optional[str] = None
        self.prev_req_type: Optional[str] = None
        self.prev_hash_ids: List[int] = []
        self.prev_completed_at: Optional[datetime] = None  # For think_time calculation

        # Models used (track all unique models in conversation)
        self.models: set = set()

        # Track shared prefix sizes (from first request)
        self.tool_tokens: int = 0
        self.system_tokens: int = 0

    def get_hash_id(self, chunk_hash: str) -> int:
        """Get or create numeric ID for a hash"""
        if chunk_hash not in self.hash_to_id:
            self.hash_to_id[chunk_hash] = self._shared_counter['next_id']
            self._shared_counter['next_id'] += 1
        return self.hash_to_id[chunk_hash]

    def create_chained_hash(self, token_ids: List[int], prev_hash: str, seq_num: int) -> str:
        """Create a hash that chains with previous block"""
        content = f"{prev_hash}:{seq_num}:" + " ".join(map(str, token_ids))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def normalize_for_cache(self, obj: Any) -> Any:
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

    def tokenize_section(self, content: Any) -> List[int]:
        """Tokenize a section (normalized)"""
        if content is None:
            return []
        normalized = self.normalize_for_cache(content)
        text = json.dumps(normalized, separators=(',', ':'))
        return self.tokenizer.encode(text, disallowed_special=())

    def get_message_content_types(self, messages: List[Dict]) -> List[str]:
        """
        Extract unique content types from all messages.

        Returns list like: ["text", "tool_use", "tool_result", "thinking"]
        """
        content_types = set()

        for msg in messages:
            content = msg.get('content', '')

            if isinstance(content, str):
                content_types.add("text")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get('type', '')
                        if block_type:
                            content_types.add(block_type)

        # Return sorted for consistency
        return sorted(content_types)

    def extract_response_info(self, response_str: Optional[str]) -> Tuple[int, str, List[str], Optional[float], Optional[str]]:
        """Extract output_tokens, stop_reason, output content types, and timing from response.

        Returns: (output_tokens, stop_reason, output_types, api_time_secs, completed_at_iso)
        """
        output_tokens = 0
        stop_reason = ""
        output_types = []
        api_time_secs = None
        completed_at_iso = None

        if not response_str:
            return output_tokens, stop_reason, output_types, api_time_secs, completed_at_iso

        try:
            resp = json.loads(response_str)
            body = resp.get('body', resp)

            usage = body.get('usage', {})
            output_tokens = usage.get('output_tokens', 0)
            stop_reason = body.get('stop_reason', '') or ''

            # Extract content types from response
            content = body.get('content', [])
            if isinstance(content, list):
                types_set = set()
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get('type', '')
                        if block_type:
                            types_set.add(block_type)
                output_types = sorted(types_set)

            # Extract timing from proxy envelope (if available)
            response_time_ms = resp.get('responseTime')
            if response_time_ms is not None:
                api_time_secs = response_time_ms / 1000.0
            completed_at_iso = resp.get('completedAt')

        except Exception:
            pass

        return output_tokens, stop_reason, output_types, api_time_secs, completed_at_iso

    def process_request(self, ts: datetime, body_str: str, response_str: Optional[str], req_type: str):
        """Process one request and add to records"""

        if self.start_ts is None:
            self.start_ts = ts

        timestamp_seconds = (ts - self.start_ts).total_seconds()

        body = json.loads(body_str)

        # Extract model
        model = body.get('model', 'unknown')
        self.models.add(model)

        # Extract and tokenize sections in order: tools → system → messages
        tools = body.get('tools', [])
        system = body.get('system', [])
        messages = body.get('messages', [])

        tools_tokens = self.tokenize_section(tools) if tools else []
        system_tokens = self.tokenize_section(system) if system else []
        messages_tokens = self.tokenize_section(messages) if messages else []

        # Capture shared prefix sizes from first request
        if not self.records:
            self.tool_tokens = len(tools_tokens)
            self.system_tokens = len(system_tokens)

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

        # Get response info (output + timing)
        output_tokens, stop_reason, output_types, api_time_secs, completed_at_iso = self.extract_response_info(response_str)

        # Compute think_time: gap between previous response completing and this request being sent
        think_time = None
        if self.prev_completed_at is not None:
            think_time = max(0.0, (ts - self.prev_completed_at).total_seconds())
        elif self.start_ts == ts:
            think_time = 0.0  # First request

        # Track completion time for next request's think_time
        if completed_at_iso:
            try:
                self.prev_completed_at = datetime.fromisoformat(completed_at_iso.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass

        # Determine what the CLIENT added this request (not cumulative history)
        req_type_short = "s" if req_type == "streaming" else ("n" if req_type == "non_streaming" else "?")

        # Check for major context reset (significant hash_id changes)
        is_major_reset = False
        if self.prev_hash_ids and hash_ids:
            # Calculate prefix preservation
            prev_set = set(self.prev_hash_ids)
            curr_set = set(hash_ids)
            preserved = len(prev_set & curr_set)
            prev_len = len(self.prev_hash_ids)
            if prev_len > 0 and preserved / prev_len < 0.5:
                is_major_reset = True

        if self.prev_req_type is None:
            # First request - user started conversation
            input_types = ["text"]
        elif is_major_reset:
            # Major context reset - show actual content types from messages
            input_types = self.get_message_content_types(messages)
        elif req_type_short == "s" and self.prev_req_type == "n":
            # Cross-turn: non-streaming -> streaming = new turn
            if self.prev_stop_reason == "tool_use":
                input_types = ["tool_result"]
            elif self.prev_stop_reason == "end_turn":
                input_types = ["text"]
            else:
                input_types = ["text"]  # Default to text for unknown
        else:
            # Within-turn pair (s->n) or other - no new client input
            input_types = []

        # Create minimal record
        record = {
            "t": round(timestamp_seconds, 1),
            "type": req_type_short,
            "model": model,
            "in": len(all_tokens),
            "out": output_tokens,
            "hash_ids": hash_ids,
            "input_types": input_types,  # What client added: ["text"], ["tool_result"], or full types on major reset
            "output_types": output_types,  # What Claude returned: ["thinking", "text", "tool_use"]
            "stop": stop_reason
        }

        # Add timing breakdown (if available from proxy response)
        if api_time_secs is not None:
            record["api_time"] = round(api_time_secs, 2)
        if think_time is not None:
            record["think_time"] = round(think_time, 2)

        # Update tracking for next request
        self.prev_stop_reason = stop_reason
        self.prev_req_type = req_type_short
        self.prev_hash_ids = hash_ids

        self.records.append(record)
        return record

    def build_trace(self, conversation_id: str) -> Dict:
        """Build final minimal trace"""
        trace = {
            "id": conversation_id[:12],
            "models": sorted(self.models),
            "block_size": self.block_size,
            "hash_id_scope": "global",
            "tool_tokens": self.tool_tokens,
            "system_tokens": self.system_tokens,
            "requests": self.records
        }
        # Store start_ts for subagent interleaving (as ISO string)
        if self.start_ts:
            trace["_start_ts"] = self.start_ts.isoformat()
        return trace


def classify_request(body_str: str) -> str:
    """Classify request type based on stream flag.

    Handles both old format (max_tokens=32000/21333) and new format
    (same max_tokens, stream=True vs stream=None).
    """
    body = json.loads(body_str)
    stream = body.get('stream')

    if stream is True:
        return 'streaming'
    elif stream is False or stream is None:
        return 'non_streaming'
    return 'unknown'


def compact_json(obj) -> str:
    """Format JSON with compact arrays on single lines"""

    def format_value(v, indent=0):
        sp = "  " * indent

        if isinstance(v, dict):
            if not v:
                return "{}"
            items = []
            for k, val in v.items():
                items.append(f'{sp}  "{k}": {format_value(val, indent + 1)}')
            return "{\n" + ",\n".join(items) + f"\n{sp}}}"

        elif isinstance(v, list):
            if not v:
                return "[]"
            if all(isinstance(x, (int, float, str, bool, type(None))) for x in v):
                return "[" + ", ".join(json.dumps(x) for x in v) + "]"
            items = [f"{sp}  {format_value(x, indent + 1)}" for x in v]
            return "[\n" + ",\n".join(items) + f"\n{sp}]"

        else:
            return json.dumps(v)

    return format_value(obj)


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


def build_message_id_map(conn) -> Dict[str, str]:
    """Build map of message_id -> request_id from database"""
    cursor = conn.cursor()
    cursor.execute("SELECT id, response FROM requests WHERE response IS NOT NULL")

    msg_id_map = {}
    for req_id, response_json in cursor.fetchall():
        try:
            response = json.loads(response_json)
            body = response.get('body', response)
            msg_id = body.get('id')
            if msg_id and msg_id.startswith('msg_'):
                msg_id_map[msg_id] = req_id
        except:
            pass

    return msg_id_map


def process_conversation(conn, conversation_id: str, jsonl_path: Path, msg_id_map: Dict[str, str], args, shared_hash_state: dict = None) -> Optional[Dict]:
    """Process a single conversation from JSONL + database"""

    # Get message IDs from JSONL
    message_ids = extract_message_ids_from_jsonl(jsonl_path)
    if not message_ids:
        return None

    # Find matching request IDs
    request_ids = []
    for msg_id in message_ids:
        if msg_id in msg_id_map:
            request_ids.append(msg_id_map[msg_id])

    if not request_ids:
        return None

    cursor = conn.cursor()

    # Get indexed requests with their data
    placeholders = ','.join('?' * len(request_ids))
    indexed_reqs = cursor.execute(f"""
        SELECT id, timestamp, body, response
        FROM requests
        WHERE id IN ({placeholders})
        ORDER BY timestamp
    """, request_ids).fetchall()

    if not indexed_reqs:
        return None

    indexed_ids = {r[0] for r in indexed_reqs}

    # Get time range with buffer
    first_ts = datetime.fromisoformat(indexed_reqs[0][1].replace('Z', '+00:00'))
    last_ts = datetime.fromisoformat(indexed_reqs[-1][1].replace('Z', '+00:00'))
    range_start = (first_ts - timedelta(seconds=300)).isoformat()
    range_end = (last_ts + timedelta(seconds=300)).isoformat()

    # Get ALL requests in time range to find streaming pairs
    all_reqs = cursor.execute("""
        SELECT id, timestamp, body, response
        FROM requests
        WHERE response IS NOT NULL
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """, (range_start, range_end)).fetchall()

    # Extract tool/system/content hashes for matching
    def extract_hashes(body_str: str) -> Tuple[str, str, str]:
        body = json.loads(body_str)
        tools = body.get('tools', [])
        system = body.get('system', [])
        messages = body.get('messages', [])
        tool_hash = hashlib.md5(json.dumps(tools, separators=(',', ':')).encode()).hexdigest()[:16]
        sys_hash = hashlib.md5(json.dumps(system, separators=(',', ':')).encode()).hexdigest()[:16]
        msg_hash = hashlib.md5(json.dumps(messages, separators=(',', ':')).encode()).hexdigest()[:16]
        return tool_hash, sys_hash, msg_hash

    # Get hashes from indexed requests
    indexed_info = []
    for req_id, ts_str, body_str, resp_str in indexed_reqs:
        tool_hash, sys_hash, msg_hash = extract_hashes(body_str)
        indexed_info.append((req_id, ts_str, tool_hash, sys_hash, msg_hash))

    # Find all conversation requests (indexed + streaming pairs)
    conversation_reqs = []
    for req_id, ts_str, body_str, resp_str in all_reqs:
        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

        if req_id in indexed_ids:
            conversation_reqs.append((req_id, ts_str, body_str, resp_str))
            continue

        # Check if this is a streaming pair (must match content, not just tools/system)
        tool_hash, sys_hash, msg_hash = extract_hashes(body_str)

        for idx_id, idx_ts, idx_tool_hash, idx_sys_hash, idx_msg_hash in indexed_info:
            idx_dt = datetime.fromisoformat(idx_ts.replace('Z', '+00:00'))
            gap = abs((ts - idx_dt).total_seconds())

            if gap < 30 and msg_hash == idx_msg_hash:
                conversation_reqs.append((req_id, ts_str, body_str, resp_str))
                break

    if len(conversation_reqs) < args.min_requests:
        return []

    # Sort by timestamp
    conversation_reqs.sort(key=lambda x: x[1])

    # Split into segments if --split-at-gap is specified
    split_gap = getattr(args, 'split_at_gap', None)
    segments = []
    current_segment = []
    prev_ts = None

    for req in conversation_reqs:
        ts = datetime.fromisoformat(req[1].replace('Z', '+00:00'))

        if split_gap and prev_ts and (ts - prev_ts).total_seconds() > split_gap:
            # Gap exceeds threshold, start new segment
            if current_segment:
                segments.append(current_segment)
            current_segment = [req]
        else:
            current_segment.append(req)

        prev_ts = ts

    if current_segment:
        segments.append(current_segment)

    # Build trace(s)
    traces = []
    for seg_idx, segment in enumerate(segments):
        if len(segment) < args.min_requests:
            continue

        builder = MinimalTraceBuilder(args.block_size, shared_hash_state=shared_hash_state)

        for req_id, ts_str, body_str, resp_str in segment:
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            req_type = classify_request(body_str)
            builder.process_request(ts, body_str, resp_str, req_type)

        # Generate trace ID with segment suffix if split
        trace = builder.build_trace(conversation_id)
        if len(segments) > 1:
            trace['id'] = f"{trace['id']}_s{seg_idx + 1}"

        trace["_analysis"] = {
            "indexed": len(indexed_ids),
            "total": len(segment),
            "unique_blocks": len(builder.hash_to_id)
        }

        traces.append(trace)

    return traces


def build_subagent_index(jsonl_dir: Path) -> Dict[str, Dict]:
    """
    Build index of all sub-agent JSONL files.

    Returns dict: agentId -> {
        'file': Path to agent-*.jsonl,
        'session_id': sub-agent's sessionId,
        'first_ts': first message timestamp,
        'last_ts': last message timestamp,
        'message_count': number of messages
    }
    """
    index = {}

    for jsonl_path in jsonl_dir.glob("agent-*.jsonl"):
        agent_id = jsonl_path.stem.replace("agent-", "")

        first_ts = None
        last_ts = None
        session_id = None
        message_count = 0

        try:
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        msg = json.loads(line.strip())
                        message_count += 1

                        # Extract agentId and sessionId from first message
                        if session_id is None:
                            session_id = msg.get('sessionId')

                        # Track timestamps
                        ts_str = msg.get('timestamp')
                        if ts_str:
                            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                            if first_ts is None or ts < first_ts:
                                first_ts = ts
                            if last_ts is None or ts > last_ts:
                                last_ts = ts
                    except:
                        pass
        except:
            continue

        if message_count > 0:
            index[agent_id] = {
                'file': jsonl_path,
                'session_id': session_id,
                'first_ts': first_ts,
                'last_ts': last_ts,
                'message_count': message_count
            }

    return index


def find_parent_subagent_links(jsonl_dir: Path) -> List[Dict]:
    """
    Find all parent -> sub-agent links by scanning toolUseResult entries.

    Returns list of dicts:
    {
        'parent_conv_id': conversation ID of parent,
        'parent_file': Path to parent JSONL,
        'tool_use_id': the tool_use_id that spawned this sub-agent,
        'agent_id': the sub-agent's ID,
        'subagent_type': type of sub-agent (e.g., 'Explore', 'general-purpose'),
        'spawn_time': timestamp when sub-agent was spawned,
        'duration_ms': total duration in milliseconds,
        'total_tokens': total tokens consumed by sub-agent,
        'tool_use_count': number of tool calls made by sub-agent,
        'status': completion status
    }
    """
    links = []

    # Only scan non-agent JSONL files (parent conversations)
    for jsonl_path in jsonl_dir.glob("*.jsonl"):
        if jsonl_path.name.startswith("agent-"):
            continue

        conv_id = jsonl_path.stem

        try:
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        msg = json.loads(line.strip())

                        # Look for toolUseResult field (appears on tool_result messages)
                        tool_result = msg.get('toolUseResult')
                        if tool_result and 'agentId' in tool_result:
                            # Extract the tool_use_id from the message content
                            tool_use_id = None
                            content = msg.get('message', {}).get('content', [])
                            if isinstance(content, list) and len(content) > 0:
                                first_block = content[0]
                                if isinstance(first_block, dict):
                                    tool_use_id = first_block.get('tool_use_id')

                            links.append({
                                'parent_conv_id': conv_id,
                                'parent_file': jsonl_path,
                                'tool_use_id': tool_use_id,
                                'agent_id': tool_result.get('agentId'),
                                'subagent_type': None,  # Will be filled from Task tool_use
                                'spawn_time': msg.get('timestamp'),
                                'duration_ms': tool_result.get('totalDurationMs'),
                                'total_tokens': tool_result.get('totalTokens'),
                                'tool_use_count': tool_result.get('totalToolUseCount'),
                                'status': tool_result.get('status', 'unknown')
                            })
                    except:
                        pass
        except:
            continue

    return links


def enrich_links_with_subagent_type(links: List[Dict], jsonl_dir: Path) -> List[Dict]:
    """
    Enrich links with subagent_type by finding the corresponding Task tool_use.

    This looks at the parent JSONL to find the Task tool_use call that matches
    each tool_use_id, and extracts the subagent_type from its input.
    """
    # Group links by parent file for efficient processing
    links_by_parent = {}
    for link in links:
        parent_file = link['parent_file']
        if parent_file not in links_by_parent:
            links_by_parent[parent_file] = []
        links_by_parent[parent_file].append(link)

    # For each parent file, find Task tool_use calls
    for parent_file, file_links in links_by_parent.items():
        tool_use_id_to_link = {l['tool_use_id']: l for l in file_links if l['tool_use_id']}

        try:
            with open(parent_file) as f:
                for line in f:
                    try:
                        msg = json.loads(line.strip())

                        # Look for assistant messages with tool_use
                        if msg.get('type') != 'assistant':
                            continue

                        content = msg.get('message', {}).get('content', [])
                        if not isinstance(content, list):
                            continue

                        for block in content:
                            if not isinstance(block, dict):
                                continue
                            if block.get('type') != 'tool_use':
                                continue
                            if block.get('name') != 'Task':
                                continue

                            tool_id = block.get('id')
                            if tool_id in tool_use_id_to_link:
                                input_data = block.get('input', {})
                                tool_use_id_to_link[tool_id]['subagent_type'] = input_data.get('subagent_type')
                    except:
                        pass
        except:
            continue

    return links


def process_subagent(conn, agent_id: str, subagent_index: Dict[str, Dict],
                     msg_id_map: Dict[str, str], args, shared_hash_state: dict = None) -> Optional[Dict]:
    """
    Process a sub-agent conversation and return its trace.

    Similar to process_conversation() but for agent-*.jsonl files.
    Returns a dict with sub-agent trace data (without the outer wrapper).
    """
    if agent_id not in subagent_index:
        return None

    agent_info = subagent_index[agent_id]
    jsonl_path = agent_info['file']

    # Get message IDs from sub-agent JSONL
    message_ids = extract_message_ids_from_jsonl(jsonl_path)
    if not message_ids:
        return None

    # Find matching request IDs
    request_ids = []
    for msg_id in message_ids:
        if msg_id in msg_id_map:
            request_ids.append(msg_id_map[msg_id])

    if not request_ids:
        return None

    cursor = conn.cursor()

    # Get indexed requests with their data
    placeholders = ','.join('?' * len(request_ids))
    indexed_reqs = cursor.execute(f"""
        SELECT id, timestamp, body, response
        FROM requests
        WHERE id IN ({placeholders})
        ORDER BY timestamp
    """, request_ids).fetchall()

    if not indexed_reqs:
        return None

    indexed_ids = {r[0] for r in indexed_reqs}

    # Get time range with buffer
    first_ts = datetime.fromisoformat(indexed_reqs[0][1].replace('Z', '+00:00'))
    last_ts = datetime.fromisoformat(indexed_reqs[-1][1].replace('Z', '+00:00'))
    range_start = (first_ts - timedelta(seconds=300)).isoformat()
    range_end = (last_ts + timedelta(seconds=300)).isoformat()

    # Get ALL requests in time range to find streaming pairs
    all_reqs = cursor.execute("""
        SELECT id, timestamp, body, response
        FROM requests
        WHERE response IS NOT NULL
          AND timestamp >= ?
          AND timestamp <= ?
        ORDER BY timestamp
    """, (range_start, range_end)).fetchall()

    # Extract tool/system/content hashes for matching
    def extract_hashes(body_str: str) -> Tuple[str, str, str]:
        body = json.loads(body_str)
        tools = body.get('tools', [])
        system = body.get('system', [])
        messages = body.get('messages', [])
        tool_hash = hashlib.md5(json.dumps(tools, separators=(',', ':')).encode()).hexdigest()[:16]
        sys_hash = hashlib.md5(json.dumps(system, separators=(',', ':')).encode()).hexdigest()[:16]
        msg_hash = hashlib.md5(json.dumps(messages, separators=(',', ':')).encode()).hexdigest()[:16]
        return tool_hash, sys_hash, msg_hash

    # Get hashes from indexed requests
    indexed_info = []
    for req_id, ts_str, body_str, resp_str in indexed_reqs:
        tool_hash, sys_hash, msg_hash = extract_hashes(body_str)
        indexed_info.append((req_id, ts_str, tool_hash, sys_hash, msg_hash))

    # Find all sub-agent requests (indexed + streaming pairs)
    subagent_reqs = []
    for req_id, ts_str, body_str, resp_str in all_reqs:
        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

        if req_id in indexed_ids:
            subagent_reqs.append((req_id, ts_str, body_str, resp_str))
            continue

        # Check if this is a streaming pair (must match content, not just tools/system)
        tool_hash, sys_hash, msg_hash = extract_hashes(body_str)

        for idx_id, idx_ts, idx_tool_hash, idx_sys_hash, idx_msg_hash in indexed_info:
            idx_dt = datetime.fromisoformat(idx_ts.replace('Z', '+00:00'))
            gap = abs((ts - idx_dt).total_seconds())

            if gap < 30 and msg_hash == idx_msg_hash:
                subagent_reqs.append((req_id, ts_str, body_str, resp_str))
                break

    if not subagent_reqs:
        return None

    # Sort by timestamp
    subagent_reqs.sort(key=lambda x: x[1])

    # Build trace for sub-agent
    builder = MinimalTraceBuilder(args.block_size, shared_hash_state=shared_hash_state)

    for req_id, ts_str, body_str, resp_str in subagent_reqs:
        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        req_type = classify_request(body_str)
        builder.process_request(ts, body_str, resp_str, req_type)

    # Return the requests list (not the full trace wrapper)
    return {
        'requests': builder.records,
        'models': sorted(builder.models),
        'tool_tokens': builder.tool_tokens,
        'system_tokens': builder.system_tokens,
        'unique_blocks': len(builder.hash_to_id)
    }


def process_conversation_with_subagents(conn, conversation_id: str, jsonl_path: Path,
                                        msg_id_map: Dict[str, str], args,
                                        subagent_index: Dict[str, Dict],
                                        parent_links: List[Dict],
                                        shared_hash_state: dict = None) -> Optional[List[Dict]]:
    """
    Process a conversation with sub-agents embedded.

    This extends process_conversation() to:
    1. Find sub-agent links for this conversation
    2. Build traces for each sub-agent
    3. Insert sub-agent traces at their spawn points in the parent trace
    """
    # First, get the parent trace using existing logic
    traces = process_conversation(conn, conversation_id, jsonl_path, msg_id_map, args, shared_hash_state=shared_hash_state)

    if not traces:
        return None

    # Find sub-agent links for this conversation
    conv_links = [l for l in parent_links if l['parent_conv_id'] == conversation_id]

    if not conv_links:
        # No sub-agents, return as-is but add totals
        for trace in traces:
            trace['totals'] = {
                'parent_tokens': {
                    'input': sum(r['in'] for r in trace['requests']),
                    'output': sum(r['out'] for r in trace['requests'])
                },
                'subagent_tokens': {'input': 0, 'output': 0},
                'combined_tokens': {
                    'input': sum(r['in'] for r in trace['requests']),
                    'output': sum(r['out'] for r in trace['requests'])
                },
                'subagent_count': 0
            }
        return traces

    # Build a map of spawn_time -> link for efficient lookup
    # Parse spawn times and sort links
    links_with_ts = []
    for link in conv_links:
        spawn_time = link.get('spawn_time')
        if spawn_time:
            try:
                spawn_ts = datetime.fromisoformat(spawn_time.replace('Z', '+00:00'))
                links_with_ts.append((spawn_ts, link))
            except:
                pass
    links_with_ts.sort(key=lambda x: x[0])

    # For each trace, insert sub-agents at appropriate positions
    for trace in traces:
        if not trace.get('requests'):
            continue

        # Get trace start time for relative timing
        trace_start_ts = None
        if trace.get('_start_ts'):
            trace_start_ts = datetime.fromisoformat(trace['_start_ts'])

        if not trace_start_ts:
            # Can't interleave without base timestamp, skip subagent processing
            continue

        # Build new requests list with sub-agents interleaved
        new_requests = []
        subagent_input_total = 0
        subagent_output_total = 0
        subagent_count = 0

        # Create a list of (timestamp, item_type, item) for sorting
        items = []

        # Add parent requests with their relative times
        for req in trace['requests']:
            items.append(('parent', req['t'], req))

        # Add sub-agents at their spawn times
        for spawn_ts, link in links_with_ts:
            agent_id = link.get('agent_id')
            if not agent_id:
                continue

            # Process the sub-agent
            subagent_trace = process_subagent(conn, agent_id, subagent_index, msg_id_map, args, shared_hash_state=shared_hash_state)
            if not subagent_trace:
                continue

            # Calculate relative time (approximate - use spawn_time from link)
            # We need to find the closest parent request to determine relative timing
            spawn_time_str = link.get('spawn_time')
            if spawn_time_str:
                try:
                    spawn_dt = datetime.fromisoformat(spawn_time_str.replace('Z', '+00:00'))
                    # Find closest parent request time
                    # For now, use the spawn time directly and hope it aligns
                    # This is a simplification - in reality we'd need the trace's base timestamp
                except:
                    continue

            # Create sub-agent entry
            subagent_entry = {
                't': 0,  # Will be set based on position
                'type': 'subagent',
                'agent_id': agent_id,
                'subagent_type': link.get('subagent_type', 'unknown'),
                'duration_ms': link.get('duration_ms', 0),
                'total_tokens': link.get('total_tokens', 0),
                'tool_use_count': link.get('tool_use_count', 0),
                'status': link.get('status', 'unknown'),
                'requests': subagent_trace['requests'],
                'models': subagent_trace['models'],
                'tool_tokens': subagent_trace['tool_tokens'],
                'system_tokens': subagent_trace['system_tokens']
            }

            # Calculate sub-agent totals
            for subreq in subagent_trace['requests']:
                subagent_input_total += subreq.get('in', 0)
                subagent_output_total += subreq.get('out', 0)
            subagent_count += 1

            items.append(('subagent', spawn_ts, subagent_entry))

        # Convert subagent spawn times to relative seconds for sorting
        # Parent requests already have relative 't' values
        items_with_relative_ts = []
        for item_type, ts, entry in items:
            if item_type == 'parent':
                # Already relative seconds
                items_with_relative_ts.append((ts, item_type, entry))
            else:
                # Convert datetime to relative seconds
                relative_ts = (ts - trace_start_ts).total_seconds()
                entry['t'] = relative_ts  # Set the subagent spawn time
                items_with_relative_ts.append((relative_ts, item_type, entry))

        # Sort all items by relative timestamp
        items_with_relative_ts.sort(key=lambda x: x[0])

        # Build final request list in chronological order
        new_requests = [entry for _, _, entry in items_with_relative_ts]

        trace['requests'] = new_requests

        # Remove internal _start_ts field (only needed for processing)
        trace.pop('_start_ts', None)

        # Add totals
        parent_input = sum(r['in'] for r in trace['requests'] if r.get('type') != 'subagent')
        parent_output = sum(r['out'] for r in trace['requests'] if r.get('type') != 'subagent')

        trace['totals'] = {
            'parent_tokens': {'input': parent_input, 'output': parent_output},
            'subagent_tokens': {'input': subagent_input_total, 'output': subagent_output_total},
            'combined_tokens': {
                'input': parent_input + subagent_input_total,
                'output': parent_output + subagent_output_total
            },
            'subagent_count': subagent_count
        }

    return traces


def main():
    parser = argparse.ArgumentParser(description='Build minimal traces from real conversations')
    parser.add_argument('db', help='Path to requests.db')
    parser.add_argument('--jsonl-dir', required=True, help='Directory containing conversation JSONL files')
    parser.add_argument('--conversation-id', help='Specific conversation ID (or omit for all)')
    parser.add_argument('--output', help='Output JSON file (single conversation)')
    parser.add_argument('--output-dir', help='Output directory (multiple conversations)')
    parser.add_argument('--block-size', type=int, default=64, help='Token block size (default: 64, only full blocks)')
    parser.add_argument('--min-requests', type=int, default=0, help='Minimum requests per conversation')
    parser.add_argument('--split-at-gap', type=int, default=None,
                        help='Split conversation when gap exceeds N seconds (e.g., 86400 for 1 day)')
    parser.add_argument('--include-subagents', action='store_true', default=False,
                        help='Include sub-agent traces nested inside parent traces')
    args = parser.parse_args()

    jsonl_dir = Path(args.jsonl_dir)
    if not jsonl_dir.exists():
        print(f"Error: JSONL directory not found: {jsonl_dir}")
        return

    conn = sqlite3.connect(args.db)

    print(f"Building message ID map from {args.db}...")
    msg_id_map = build_message_id_map(conn)
    print(f"  Found {len(msg_id_map):,} message IDs")
    print()

    # Build sub-agent index if requested
    subagent_index = {}
    parent_links = []
    if args.include_subagents:
        print("Building sub-agent index...")
        subagent_index = build_subagent_index(jsonl_dir)
        print(f"  Found {len(subagent_index)} sub-agent conversations")

        print("Finding parent -> sub-agent links...")
        parent_links = find_parent_subagent_links(jsonl_dir)
        parent_links = enrich_links_with_subagent_type(parent_links, jsonl_dir)
        print(f"  Found {len(parent_links)} sub-agent links")
        print()

    # Shared hash state across all conversations in this batch run
    # Makes hash_ids globally consistent: same content = same ID
    shared_hash_state = {'hash_to_id': {}, 'next_id': 1}

    # Find JSONL files (exclude agent-* files from main processing)
    jsonl_files = [f for f in jsonl_dir.glob("*.jsonl") if not f.name.startswith("agent-")]
    print(f"Found {len(jsonl_files)} parent conversation JSONL files in {jsonl_dir}")
    if args.include_subagents:
        agent_files = list(jsonl_dir.glob("agent-*.jsonl"))
        print(f"Found {len(agent_files)} sub-agent JSONL files")
    print()

    if args.conversation_id:
        # Single conversation
        jsonl_path = jsonl_dir / f"{args.conversation_id}.jsonl"
        if not jsonl_path.exists():
            print(f"Error: JSONL file not found: {jsonl_path}")
            return

        print(f"Processing: {args.conversation_id}")
        if args.include_subagents:
            traces = process_conversation_with_subagents(
                conn, args.conversation_id, jsonl_path, msg_id_map, args,
                subagent_index, parent_links, shared_hash_state=shared_hash_state
            )
        else:
            traces = process_conversation(conn, args.conversation_id, jsonl_path, msg_id_map, args, shared_hash_state=shared_hash_state)

        if traces:
            for i, trace in enumerate(traces):
                trace_id = trace['id']
                if args.output:
                    # For splits, add segment number to filename
                    if len(traces) > 1:
                        base, ext = os.path.splitext(args.output)
                        output_path = f"{base}_s{i+1}{ext}"
                    else:
                        output_path = args.output
                else:
                    output_path = f"trace_{trace_id}.json"
                with open(output_path, 'w') as f:
                    f.write(compact_json(trace))
                print(f"Written: {output_path}")
                print(f"  Requests: {len(trace['requests'])}")
                print(f"  Unique blocks: {trace['_analysis']['unique_blocks']}")
        else:
            print("  No matching requests found")
    else:
        # All conversations
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)

        processed = 0
        skipped = 0
        subagent_total = 0

        for jsonl_path in sorted(jsonl_files):
            conv_id = jsonl_path.stem
            if args.include_subagents:
                traces = process_conversation_with_subagents(
                    conn, conv_id, jsonl_path, msg_id_map, args,
                    subagent_index, parent_links, shared_hash_state=shared_hash_state
                )
            else:
                traces = process_conversation(conn, conv_id, jsonl_path, msg_id_map, args, shared_hash_state=shared_hash_state)

            if traces:
                for trace in traces:
                    processed += 1
                    trace_id = trace['id']
                    if args.output_dir:
                        output_path = os.path.join(args.output_dir, f"{trace_id[:16]}.json")
                        with open(output_path, 'w') as f:
                            f.write(compact_json(trace))

                    # Count sub-agents in trace
                    trace_subagents = sum(1 for r in trace.get('requests', []) if r.get('type') == 'subagent')
                    subagent_total += trace_subagents

                    if args.include_subagents and trace_subagents > 0:
                        print(f"  {trace_id}: {len(trace['requests'])} items ({trace_subagents} subagents), {trace['_analysis']['unique_blocks']} blocks")
                    else:
                        print(f"  {trace_id}: {len(trace['requests'])} requests, {trace['_analysis']['unique_blocks']} blocks")
            else:
                skipped += 1

        print()
        print(f"Processed: {processed} conversations")
        if args.include_subagents:
            print(f"Total sub-agents included: {subagent_total}")
        print(f"Skipped: {skipped} (no matching requests or < {args.min_requests} requests)")
        if args.output_dir:
            print(f"Output: {args.output_dir}/")

    conn.close()


if __name__ == "__main__":
    main()
