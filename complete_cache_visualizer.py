#!/usr/bin/env python3
"""
Complete Cache Visualizer

Analyzes cache behavior using COMPLETE conversation (streaming + non-streaming).
Creates visualizations and detailed statistics with TTL analysis.

Version: 21.0 (2024-10-31)
IMPORTANT: Now normalizes cache_control markers for accurate cache simulation
NEW: Chunk prefill count analysis (formerly churn rate)
"""

import sqlite3
import json
import hashlib
import tiktoken
import time
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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
class MessageSegment:
    """A single message/block in the request"""
    type: str  # 'system', 'user_text', 'assistant_text', 'tool_use', 'tool_result'
    tokens: int
    preview: str  # First 300 chars
    start_token: int  # Starting token position in request
    end_token: int    # Ending token position in request
    is_cache_hit: bool = False  # Whether this segment was a cache hit

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
    output_tokens: int
    api_cache_read: Optional[int]
    api_cache_creation: Optional[int]
    api_input_tokens: Optional[int]
    model: Optional[str]
    # Message breakdown
    user_text_messages: int = 0
    tool_use_messages: int = 0
    tool_result_messages: int = 0
    assistant_messages: int = 0
    # Message sequence
    message_segments: List['MessageSegment'] = None

class CompleteCacheSimulator:
    """Cache simulator that processes ALL requests"""

    def __init__(self, block_size: int = 64, ttl: int = 300):
        self.block_size = block_size
        self.ttl = ttl
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        self.cache: Dict[str, Dict[int, CacheChunk]] = {}
        self.analyses: List[RequestAnalysis] = []
        self.cached_requests: List[CachedRequest] = []  # For TTL analysis

    def normalize_body(self, body):
        """Remove cache_control markers recursively for cache key calculation"""
        import copy
        normalized = copy.deepcopy(body)

        def remove_cache_control(obj):
            """Recursively remove cache_control from nested structures"""
            if isinstance(obj, dict):
                # Remove cache_control if present
                obj.pop('cache_control', None)
                # Recurse into nested objects
                for value in obj.values():
                    remove_cache_control(value)
            elif isinstance(obj, list):
                for item in obj:
                    remove_cache_control(item)

        remove_cache_control(normalized)
        return normalized

    def extract_all_tokens(self, body):
        """Extract all tokens from request (normalized - cache_control removed)"""
        # Normalize body to remove cache_control markers
        normalized_body = self.normalize_body(body)

        all_tokens = []

        if 'tools' in normalized_body and normalized_body['tools']:
            all_tokens.extend(self.tokenizer.encode(json.dumps(normalized_body['tools'], separators=(',', ':'))))

        if 'system' in normalized_body:
            all_tokens.extend(self.tokenizer.encode(json.dumps(normalized_body['system'], separators=(',', ':'))))

        if 'messages' in normalized_body and normalized_body['messages']:
            all_tokens.extend(self.tokenizer.encode(json.dumps(normalized_body['messages'], separators=(',', ':'))))

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

        # Extract model
        model = body.get('model', 'unknown')

        # Extract API metrics and output tokens
        api_cache_read = None
        api_cache_creation = None
        api_input_tokens = None
        output_tokens = 0

        if response_str:
            try:
                resp = json.loads(response_str)
                usage = resp.get('body', {}).get('usage') if 'body' in resp else resp.get('usage', {})
                if usage:
                    api_cache_read = usage.get('cache_read_input_tokens')
                    api_cache_creation = usage.get('cache_creation_input_tokens')
                    api_input_tokens = usage.get('input_tokens')
                    output_tokens = usage.get('output_tokens', 0)
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

        # Analyze message breakdown
        user_text, tool_use, tool_result, assistant = analyze_message_breakdown(body)

        # Extract message segments for detailed visualization
        message_segments = extract_message_segments(body, self.tokenizer)

        # Mark which segments are cache hits
        for segment in message_segments:
            # A segment is a cache hit if it's fully within the cache read range
            if segment.end_token <= cache_read_tokens:
                segment.is_cache_hit = True

        # Store for TTL analysis
        self.cached_requests.append(CachedRequest(
            timestamp=ts,
            chunks=chunks,
            total_tokens=len(all_tokens)
        ))

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
            output_tokens=output_tokens,
            api_cache_read=api_cache_read,
            api_cache_creation=api_cache_creation,
            api_input_tokens=api_input_tokens,
            model=model,
            user_text_messages=user_text,
            tool_use_messages=tool_use,
            tool_result_messages=tool_result,
            assistant_messages=assistant,
            message_segments=message_segments
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

def format_preview_text(text: str, max_line_length: int = 100) -> str:
    """Format preview text preserving original newlines but wrapping long lines"""
    lines = text.split('\n')
    formatted_lines = []

    for line in lines:
        if len(line) <= max_line_length:
            formatted_lines.append(line)
        else:
            # Wrap long lines
            for i in range(0, len(line), max_line_length):
                formatted_lines.append(line[i:i+max_line_length])

    return '<br>'.join(formatted_lines)

def analyze_message_breakdown(body: dict) -> tuple:
    """Analyze message types in request body

    Returns: (user_text_messages, tool_use_messages, tool_result_messages, assistant_messages)
    """
    user_text = 0
    tool_use = 0
    tool_result = 0
    assistant = 0

    messages = body.get('messages', [])
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'user':
            # Check if it's a tool result or regular text
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get('type') == 'tool_result':
                            tool_result += 1
                        elif block.get('type') == 'text':
                            user_text += 1
            elif isinstance(content, str):
                user_text += 1
        elif role == 'assistant':
            # Check if it contains tool use
            if isinstance(content, list):
                has_tool_use = any(isinstance(b, dict) and b.get('type') == 'tool_use' for b in content)
                if has_tool_use:
                    tool_use += 1
                else:
                    assistant += 1
            else:
                assistant += 1

    return (user_text, tool_use, tool_result, assistant)

def normalize_for_cache(body: dict) -> dict:
    """Remove cache_control markers recursively"""
    import copy
    normalized = copy.deepcopy(body)

    def remove_cache_control(obj):
        if isinstance(obj, dict):
            obj.pop('cache_control', None)
            for value in obj.values():
                remove_cache_control(value)
        elif isinstance(obj, list):
            for item in obj:
                remove_cache_control(item)

    remove_cache_control(normalized)
    return normalized

def extract_message_segments(body: dict, tokenizer) -> List[MessageSegment]:
    """Extract detailed message sequence from request body with token positions

    Note: Token positions follow the cache hierarchy: tools → system → messages
    Note: Normalizes out cache_control for consistent hashing
    """
    # Normalize body to remove cache_control
    body = normalize_for_cache(body)

    segments = []
    current_position = 0

    # Tools (come first in cache hierarchy)
    if 'tools' in body and body['tools']:
        tools_content = json.dumps(body['tools'], separators=(',', ':'))
        tokens = tokenizer.encode(tools_content)
        num_tokens = len(tokens)

        segments.append(MessageSegment(
            type='tools',
            tokens=num_tokens,
            preview=f"{len(body['tools'])} tools defined",
            start_token=current_position,
            end_token=current_position + num_tokens
        ))
        current_position += num_tokens

    # System prompt (comes second in cache hierarchy)
    if 'system' in body and body['system']:
        system_content = json.dumps(body['system'], separators=(',', ':'))
        tokens = tokenizer.encode(system_content)
        num_tokens = len(tokens)

        # Get preview
        if isinstance(body['system'], list):
            preview_text = ' '.join([block.get('text', '')[:5000] for block in body['system'] if isinstance(block, dict)])[:5000]
        else:
            preview_text = str(body['system'])[:5000]

        segments.append(MessageSegment(
            type='system',
            tokens=num_tokens,
            preview=preview_text,
            start_token=current_position,
            end_token=current_position + num_tokens
        ))
        current_position += num_tokens

    # Messages in sequence
    messages = body.get('messages', [])
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        # Tokenize this message
        msg_json = json.dumps(msg, separators=(',', ':'))
        tokens = tokenizer.encode(msg_json)

        if role == 'user':
            # Check if it contains tool results or text
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get('type', 'unknown')
                        block_json = json.dumps(block, separators=(',', ':'))
                        block_tokens = tokenizer.encode(block_json)

                        if block_type == 'tool_result':
                            preview = str(block.get('content', ''))[:5000]
                            num_tokens = len(block_tokens)
                            segments.append(MessageSegment(
                                type='tool_result',
                                tokens=num_tokens,
                                preview=preview,
                                start_token=current_position,
                                end_token=current_position + num_tokens
                            ))
                            current_position += num_tokens
                        elif block_type == 'text':
                            preview = block.get('text', '')[:5000]
                            num_tokens = len(block_tokens)
                            segments.append(MessageSegment(
                                type='user_text',
                                tokens=num_tokens,
                                preview=preview,
                                start_token=current_position,
                                end_token=current_position + num_tokens
                            ))
                            current_position += num_tokens
            elif isinstance(content, str):
                preview = content[:5000]
                num_tokens = len(tokens)
                segments.append(MessageSegment(
                    type='user_text',
                    tokens=num_tokens,
                    preview=preview,
                    start_token=current_position,
                    end_token=current_position + num_tokens
                ))
                current_position += num_tokens

        elif role == 'assistant':
            # Check if it contains tool use or text
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get('type', 'unknown')
                        block_json = json.dumps(block, separators=(',', ':'))
                        block_tokens = tokenizer.encode(block_json)

                        if block_type == 'tool_use':
                            tool_name = block.get('name', 'unknown')
                            preview = f"Tool: {tool_name}, Input: {str(block.get('input', {}))[:4900]}"
                            num_tokens = len(block_tokens)
                            segments.append(MessageSegment(
                                type='tool_use',
                                tokens=num_tokens,
                                preview=preview,
                                start_token=current_position,
                                end_token=current_position + num_tokens
                            ))
                            current_position += num_tokens
                        elif block_type == 'text':
                            preview = block.get('text', '')[:5000]
                            num_tokens = len(block_tokens)
                            segments.append(MessageSegment(
                                type='assistant_text',
                                tokens=num_tokens,
                                preview=preview,
                                start_token=current_position,
                                end_token=current_position + num_tokens
                            ))
                            current_position += num_tokens
            elif isinstance(content, str):
                preview = content[:5000]
                num_tokens = len(tokens)
                segments.append(MessageSegment(
                    type='assistant_text',
                    tokens=num_tokens,
                    preview=preview,
                    start_token=current_position,
                    end_token=current_position + num_tokens
                ))
                current_position += num_tokens

    return segments

def calculate_time_stats(analyses: List[RequestAnalysis]) -> Dict:
    """Calculate time between turns statistics"""
    if len(analyses) < 2:
        return None

    gaps = []
    for i in range(1, len(analyses)):
        gap = (analyses[i].timestamp - analyses[i-1].timestamp).total_seconds()
        gaps.append(gap)

    gaps_sorted = sorted(gaps)
    n = len(gaps_sorted)

    return {
        'mean': np.mean(gaps),
        'p25': gaps_sorted[int(n * 0.25)],
        'p50': gaps_sorted[int(n * 0.50)],
        'p75': gaps_sorted[int(n * 0.75)],
        'p90': gaps_sorted[int(n * 0.90)],
    }

def calculate_model_stats(analyses: List[RequestAnalysis]) -> Dict[str, Dict]:
    """Calculate usage statistics by model"""
    model_stats = {}

    for a in analyses:
        model = a.model or 'unknown'
        if model not in model_stats:
            model_stats[model] = {
                'requests': 0,
                'input_tokens': 0,
                'output_tokens': 0
            }

        model_stats[model]['requests'] += 1
        model_stats[model]['input_tokens'] += a.total_tokens
        model_stats[model]['output_tokens'] += a.output_tokens

    return model_stats

def calculate_context_stats(analyses: List[RequestAnalysis]) -> Dict:
    """Calculate context length statistics"""
    if not analyses:
        return None

    context_lengths = [a.total_tokens for a in analyses]
    context_sorted = sorted(context_lengths)
    n = len(context_sorted)

    # Calculate incremental changes (message additions)
    incremental_changes = []
    for i in range(1, len(analyses)):
        change = analyses[i].total_tokens - analyses[i-1].total_tokens
        if change > 0:  # Only count additions
            incremental_changes.append(change)

    inc_sorted = sorted(incremental_changes) if incremental_changes else []
    n_inc = len(inc_sorted)

    return {
        'mean': np.mean(context_lengths),
        'p25': context_sorted[int(n * 0.25)],
        'p50': context_sorted[int(n * 0.50)],
        'p75': context_sorted[int(n * 0.75)],
        'p90': context_sorted[int(n * 0.90)],
        'max': context_sorted[-1],
        'incremental': {
            'mean': np.mean(incremental_changes) if incremental_changes else 0,
            'p25': inc_sorted[int(n_inc * 0.25)] if inc_sorted else 0,
            'p50': inc_sorted[int(n_inc * 0.50)] if inc_sorted else 0,
            'p75': inc_sorted[int(n_inc * 0.75)] if inc_sorted else 0,
            'p90': inc_sorted[int(n_inc * 0.90)] if inc_sorted else 0,
            'max': inc_sorted[-1] if inc_sorted else 0,
        }
    }

@dataclass
class CachedRequest:
    """Pre-tokenized request for TTL analysis"""
    timestamp: datetime
    chunks: List[CacheChunk]
    total_tokens: int

def analyze_ttl_impact_fast(cached_requests: List[CachedRequest], block_size: int, interpolation_interval: int = 15, skip_long_ttls: bool = False, show_progress: bool = True) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, List[Tuple[datetime, int]]], Dict[int, List[Tuple[datetime, float]]], Dict[int, float], Dict[int, float]]:
    """Fast TTL analysis using pre-tokenized requests with interpolated time points

    Returns:
        - hit_rates: Dict mapping TTL to cache hit rate
        - max_working_set: Dict mapping TTL to maximum working set size in tokens
        - working_set_timeseries: Dict mapping TTL to list of (timestamp, tokens_in_cache) - only for graphed TTLs
        - cache_hit_rate_timeseries: Dict mapping TTL to list of (timestamp, hit_rate%) - only for graphed TTLs
        - churn_rates: Dict mapping TTL to average churn rate
        - churn_efficiency: Dict mapping TTL to churn efficiency %
    """

    # All TTLs for statistics
    if skip_long_ttls:
        all_ttl_values = [60, 120, 180, 240, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400]
    else:
        all_ttl_values = [60, 120, 180, 240, 300, 600, 900, 1800, 3600, 7200, 14400, 28800, 86400, 259200]

    # Only generate detailed timeseries for graphed TTLs (optimization)
    graphed_ttls = [60, 300, 900, 3600, 86400]

    results = {}
    max_working_set = {}
    working_set_timeseries = {}
    cache_hit_rate_timeseries = {}
    churn_rates = {}
    churn_efficiency = {}

    if not cached_requests:
        return results, max_working_set, working_set_timeseries, cache_hit_rate_timeseries, churn_rates, churn_efficiency

    # Get time range for interpolation
    start_time = cached_requests[0].timestamp
    end_time = cached_requests[-1].timestamp + timedelta(minutes=interpolation_interval)  # Buffer for decay

    total_ttls = len(all_ttl_values)

    for ttl_idx, ttl in enumerate(all_ttl_values):
        # Progress indicator
        if show_progress:
            ttl_labels = {60: '1m', 120: '2m', 180: '3m', 240: '4m', 300: '5m',
                         600: '10m', 900: '15m', 1800: '30m', 3600: '1h',
                         7200: '2h', 14400: '4h', 28800: '8h', 86400: '24h', 259200: '72h'}
            print(f"\r  Analyzing TTL impact... {ttl_idx+1}/{total_ttls} ({ttl_labels.get(ttl, str(ttl)+'s')})", end='', flush=True)
        use_interpolation = ttl in graphed_ttls
        cache: Dict[str, Dict[int, CacheChunk]] = {}
        total_cache_read = 0
        total_tokens = 0
        max_cache_tokens = 0

        # Track chunk lifecycle for churn analysis
        chunk_lifecycle = {}  # (hash, seq) -> creation_count

        # Generate time points
        if use_interpolation:
            # Interpolated time points using specified interval
            time_points = []
            current_time = start_time
            while current_time <= end_time:
                time_points.append(current_time)
                current_time += timedelta(minutes=interpolation_interval)

            # Add actual request timestamps
            for req in cached_requests:
                if req.timestamp not in time_points:
                    time_points.append(req.timestamp)

            time_points.sort()
        else:
            # Just use request timestamps (faster)
            time_points = [req.timestamp for req in cached_requests]

        # Process requests and calculate working set at each time point
        req_index = 0
        timeseries = [] if use_interpolation else None
        hit_rate_timeseries = [] if use_interpolation else None
        cumulative_cache_read = 0
        cumulative_total_tokens = 0

        for time_point in time_points:
            # Process all requests up to this time point
            while req_index < len(cached_requests) and cached_requests[req_index].timestamp <= time_point:
                req = cached_requests[req_index]
                total_tokens += req.total_tokens

                # Find prefix match
                matched_tokens = 0
                matched_chunks = 0

                for chunk in req.chunks:
                    if chunk.chunk_hash in cache:
                        cached_chunks = cache[chunk.chunk_hash]
                        if chunk.sequence_number in cached_chunks:
                            cached_chunk = cached_chunks[chunk.sequence_number]
                            time_diff = (req.timestamp - cached_chunk.last_accessed).total_seconds()
                            if time_diff <= ttl:
                                matched_tokens += chunk.token_count
                                matched_chunks += 1
                                # Refresh
                                cached_chunks[chunk.sequence_number] = CacheChunk(
                                    token_count=chunk.token_count,
                                    chunk_hash=chunk.chunk_hash,
                                    sequence_number=chunk.sequence_number,
                                    created_at=cached_chunk.created_at,
                                    last_accessed=req.timestamp
                                )
                            else:
                                break
                        else:
                            break
                    else:
                        break

                total_cache_read += matched_tokens
                cumulative_cache_read += matched_tokens
                cumulative_total_tokens += req.total_tokens

                # Update cache with new chunks
                for chunk in req.chunks[matched_chunks:]:
                    # Track chunk creation for churn analysis
                    chunk_key = (chunk.chunk_hash, chunk.sequence_number)
                    if chunk_key not in chunk_lifecycle:
                        chunk_lifecycle[chunk_key] = 1
                    else:
                        chunk_lifecycle[chunk_key] += 1

                    if chunk.chunk_hash not in cache:
                        cache[chunk.chunk_hash] = {}
                    cache[chunk.chunk_hash][chunk.sequence_number] = CacheChunk(
                        token_count=chunk.token_count,
                        chunk_hash=chunk.chunk_hash,
                        sequence_number=chunk.sequence_number,
                        created_at=req.timestamp,
                        last_accessed=req.timestamp
                    )

                req_index += 1

            # Calculate working set size at this time point
            current_cache_tokens = 0
            for chunk_hash_dict in cache.values():
                for chunk in chunk_hash_dict.values():
                    time_since_access = (time_point - chunk.last_accessed).total_seconds()
                    if time_since_access <= ttl:
                        current_cache_tokens += chunk.token_count

            # Calculate rolling cache hit rate
            rolling_hit_rate = (cumulative_cache_read / cumulative_total_tokens * 100) if cumulative_total_tokens > 0 else 0.0

            if use_interpolation:
                timeseries.append((time_point, current_cache_tokens))
                hit_rate_timeseries.append((time_point, rolling_hit_rate))
            max_cache_tokens = max(max_cache_tokens, current_cache_tokens)

        hit_rate = total_cache_read / total_tokens if total_tokens > 0 else 0.0
        results[ttl] = hit_rate
        max_working_set[ttl] = max_cache_tokens
        if use_interpolation:
            working_set_timeseries[ttl] = timeseries
            cache_hit_rate_timeseries[ttl] = hit_rate_timeseries

        # Calculate churn metrics
        if chunk_lifecycle:
            creation_counts = list(chunk_lifecycle.values())
            avg_churn = sum(creation_counts) / len(creation_counts)
            stable_chunks = sum(1 for count in creation_counts if count == 1)
            efficiency = (stable_chunks / len(creation_counts)) * 100
        else:
            avg_churn = 0.0
            efficiency = 0.0

        churn_rates[ttl] = avg_churn
        churn_efficiency[ttl] = efficiency

    # Final progress message
    if show_progress:
        print(f"\r  Analyzing TTL impact... {total_ttls}/{total_ttls} (complete)                                    ", flush=True)

    return results, max_working_set, working_set_timeseries, cache_hit_rate_timeseries, churn_rates, churn_efficiency

def create_cache_visualization(analyses: List[RequestAnalysis]) -> go.Figure:
    """Create stacked bar chart of cache usage"""

    # Subsample if too many requests for visualization
    if len(analyses) > 5000:
        step = len(analyses) // 5000
        sampled = analyses[::step]
        note = f" (showing every {step}th request)"
    else:
        sampled = analyses
        note = ""

    fig = go.Figure()
    x_labels = [f"R{a.index}" for a in sampled]

    # Cache read (grey)
    cache_read = [a.cache_read_tokens for a in sampled]
    fig.add_trace(go.Bar(
        x=x_labels, y=cache_read,
        name='Cache Read',
        marker=dict(color='lightgrey'),
        opacity=0.8
    ))

    # Cache creation (red)
    cache_creation = [a.cache_creation_tokens for a in sampled]
    fig.add_trace(go.Bar(
        x=x_labels, y=cache_creation,
        name='Cache Creation',
        marker=dict(color='lightcoral'),
        opacity=0.8
    ))

    # Output (blue)
    output = [a.output_tokens for a in sampled]
    fig.add_trace(go.Bar(
        x=x_labels, y=output,
        name='Output',
        marker=dict(color='darkblue'),
        opacity=0.9
    ))

    num_requests = len(sampled)
    width = max(1200, num_requests * 40)

    title = f'Cache Usage - {len(analyses):,} requests{note}' if len(analyses) != num_requests else None

    fig.update_layout(
        title=title,
        xaxis_title='Request Number',
        yaxis_title='Tokens',
        barmode='stack',
        height=700,
        width=width,
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(showticklabels=False)
    )

    return fig

def create_cache_visualization_skinny(analyses: List[RequestAnalysis]) -> go.Figure:
    """Create compressed/skinny stacked bar chart with no gaps - fits screen width"""

    fig = go.Figure()
    x_labels = [f"R{a.index}" for a in analyses]

    # Cache read (grey)
    cache_read = [a.cache_read_tokens for a in analyses]
    fig.add_trace(go.Bar(
        x=x_labels, y=cache_read,
        name='Cache Read',
        marker=dict(color='lightgrey'),
        opacity=0.8
    ))

    # Cache creation (red)
    cache_creation = [a.cache_creation_tokens for a in analyses]
    fig.add_trace(go.Bar(
        x=x_labels, y=cache_creation,
        name='Cache Creation',
        marker=dict(color='lightcoral'),
        opacity=0.8
    ))

    # Output (blue)
    output = [a.output_tokens for a in analyses]
    fig.add_trace(go.Bar(
        x=x_labels, y=output,
        name='Output',
        marker=dict(color='darkblue'),
        opacity=0.9
    ))

    fig.update_layout(
        title=f'Cache Usage (Skinny) - {len(analyses):,} requests',
        xaxis_title='Request Number',
        yaxis_title='Tokens',
        barmode='stack',
        height=700,
        width=1920,  # Fixed width for screen
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(showticklabels=False),
        bargap=0,  # NO GAP between bars
        bargroupgap=0  # NO GAP between groups
    )

    return fig

def create_detailed_cache_visualization(analyses: List[RequestAnalysis]) -> go.Figure:
    """Create detailed stacked bar chart showing actual message sequence with cache hit coloring"""

    # Subsample if too many requests
    if len(analyses) > 5000:
        step = len(analyses) // 5000
        sampled = analyses[::step]
        note = f" (showing every {step}th request)"
    else:
        sampled = analyses
        note = ""

    # Color mapping for different message types (when cache hit)
    cache_hit_colors = {
        'tools': '#B8B8B8',           # Medium-light grey
        'system': '#D0D0D0',          # Light grey
        'user_text': '#E8E8E8',       # Very light grey
        'assistant_text': '#808080',  # Medium grey
        'tool_use': '#606060',        # Dark grey
        'tool_result': '#A8A8A8'      # Medium-light grey
    }

    # Color for cache misses (same as cache creation in simple graph)
    cache_miss_color = 'lightcoral'

    fig = go.Figure()

    # For each request, we need to create trace data for each segment
    max_segments = max(len(a.message_segments) if a.message_segments else 0 for a in sampled)

    # Create traces for each segment position
    for seg_idx in range(max_segments):
        # Collect data for this segment position across all requests
        y_values = []
        hover_texts = []
        colors_list = []

        for analysis in sampled:
            if analysis.message_segments and seg_idx < len(analysis.message_segments):
                segment = analysis.message_segments[seg_idx]
                y_values.append(segment.tokens)

                # Determine color based on cache hit status
                if segment.is_cache_hit:
                    color = cache_hit_colors.get(segment.type, '#999999')
                    status = "CACHE HIT"
                else:
                    color = cache_miss_color
                    status = "CACHE MISS"

                # Format preview preserving original line breaks and wrapping long lines
                preview_text = segment.preview[:5000]  # ~1000 words
                formatted_preview = format_preview_text(preview_text, max_line_length=100)
                hover_text = f"<b>{status}</b><br>Type: {segment.type}<br>Tokens: {segment.tokens:,}<br><br><b>Preview:</b><br>{formatted_preview}"
                hover_texts.append(hover_text)
                colors_list.append(color)
            else:
                y_values.append(0)
                hover_texts.append("")
                colors_list.append('#FFFFFF')

        fig.add_trace(go.Bar(
            x=[f"R{a.index}" for a in sampled],
            y=y_values,
            name=f'Segment {seg_idx}',
            marker=dict(color=colors_list, line=dict(width=0)),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False
        ))

    # Add output tokens (same color as simple graph)
    output = [a.output_tokens for a in sampled]
    fig.add_trace(go.Bar(
        x=[f"R{a.index}" for a in sampled],
        y=output,
        name='Output',
        marker=dict(color='darkblue'),
        opacity=0.9
    ))

    # Add legend-only entries to explain colors
    for msg_type, color in cache_hit_colors.items():
        fig.add_trace(go.Bar(
            x=['LEGEND_ONLY'],
            y=[1],
            marker=dict(color=color),
            name=f'{msg_type} (cached)',
            showlegend=True,
            visible='legendonly'
        ))
    fig.add_trace(go.Bar(
        x=['LEGEND_ONLY'],
        y=[1],
        marker=dict(color=cache_miss_color),
        name='Cache Miss',
        showlegend=True,
        visible='legendonly'
    ))

    num_requests = len(sampled)
    width = max(1200, num_requests * 40)

    title = f'Detailed Message Sequence - {len(analyses):,} requests{note}' if len(analyses) != num_requests else f'Detailed Message Sequence - {len(analyses):,} requests'

    fig.update_layout(
        title=title,
        xaxis_title='Request Number',
        yaxis_title='Tokens',
        barmode='stack',
        height=700,
        width=width,
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(showticklabels=False)
    )

    return fig

def create_detailed_cache_visualization_skinny(analyses: List[RequestAnalysis]) -> go.Figure:
    """Create detailed skinny stacked bar chart showing actual message sequence with cache hit coloring - no gaps"""

    # Color mapping for different message types (when cache hit)
    cache_hit_colors = {
        'tools': '#B8B8B8',           # Medium-light grey
        'system': '#D0D0D0',          # Light grey
        'user_text': '#E8E8E8',       # Very light grey
        'assistant_text': '#808080',  # Medium grey
        'tool_use': '#606060',        # Dark grey
        'tool_result': '#A8A8A8'      # Medium-light grey
    }

    # Color for cache misses (same as cache creation in simple graph)
    cache_miss_color = 'lightcoral'

    fig = go.Figure()

    # For each request, we need to create trace data for each segment
    max_segments = max(len(a.message_segments) if a.message_segments else 0 for a in analyses)

    # Create traces for each segment position
    for seg_idx in range(max_segments):
        # Collect data for this segment position across all requests
        y_values = []
        hover_texts = []
        colors_list = []

        for analysis in analyses:
            if analysis.message_segments and seg_idx < len(analysis.message_segments):
                segment = analysis.message_segments[seg_idx]
                y_values.append(segment.tokens)

                # Determine color based on cache hit status
                if segment.is_cache_hit:
                    color = cache_hit_colors.get(segment.type, '#999999')
                    status = "CACHE HIT"
                else:
                    color = cache_miss_color
                    status = "CACHE MISS"

                # Format preview preserving original line breaks and wrapping long lines
                preview_text = segment.preview[:5000]  # ~1000 words
                formatted_preview = format_preview_text(preview_text, max_line_length=100)
                hover_text = f"<b>{status}</b><br>Type: {segment.type}<br>Tokens: {segment.tokens:,}<br><br><b>Preview:</b><br>{formatted_preview}"
                hover_texts.append(hover_text)
                colors_list.append(color)
            else:
                y_values.append(0)
                hover_texts.append("")
                colors_list.append('#FFFFFF')

        fig.add_trace(go.Bar(
            x=[f"R{a.index}" for a in analyses],
            y=y_values,
            name=f'Segment {seg_idx}',
            marker=dict(color=colors_list, line=dict(width=0)),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False
        ))

    # Add output tokens (same color as simple graph)
    output = [a.output_tokens for a in analyses]
    fig.add_trace(go.Bar(
        x=[f"R{a.index}" for a in analyses],
        y=output,
        name='Output',
        marker=dict(color='darkblue'),
        opacity=0.9
    ))

    # Add legend-only entries to explain colors
    for msg_type, color in cache_hit_colors.items():
        fig.add_trace(go.Bar(
            x=['LEGEND_ONLY'],
            y=[1],
            marker=dict(color=color),
            name=f'{msg_type} (cached)',
            showlegend=True,
            visible='legendonly'
        ))
    fig.add_trace(go.Bar(
        x=['LEGEND_ONLY'],
        y=[1],
        marker=dict(color=cache_miss_color),
        name='Cache Miss',
        showlegend=True,
        visible='legendonly'
    ))

    fig.update_layout(
        title=f'Detailed Message Sequence (Skinny) - {len(analyses):,} requests',
        xaxis_title='Request Number',
        yaxis_title='Tokens',
        barmode='stack',
        height=700,
        width=1920,  # Fixed width for screen
        showlegend=True,
        plot_bgcolor='white',
        xaxis=dict(showticklabels=False),
        bargap=0,  # NO GAP between bars
        bargroupgap=0  # NO GAP between groups
    )

    return fig

def create_ttl_graph(ttl_results: Dict[int, float]) -> go.Figure:
    """Create graph showing cache hit rate vs TTL"""

    ttls = sorted(ttl_results.keys())
    hit_rates = [ttl_results[ttl] * 100 for ttl in ttls]

    # Convert seconds to readable labels
    labels = []
    for ttl in ttls:
        if ttl < 60:
            labels.append(f"{ttl}s")
        elif ttl < 3600:
            labels.append(f"{ttl//60}m")
        elif ttl < 86400:
            labels.append(f"{ttl//3600}h")
        else:
            labels.append(f"{ttl//86400}d")

    # Calculate dynamic y-axis minimum (round down to nearest 10%)
    min_hit_rate = min(hit_rates)
    y_min = int(min_hit_rate / 10) * 10  # Round down to nearest 10%

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=labels,
        y=hit_rates,
        mode='lines+markers',
        marker=dict(size=8, color='darkblue'),
        line=dict(width=2, color='darkblue'),
        name='Cache Hit Rate'
    ))

    fig.update_layout(
        title='Cache Hit Rate vs TTL',
        xaxis_title='TTL',
        yaxis_title='Cache Hit Rate (%)',
        height=500,
        width=1000,
        plot_bgcolor='white',
        yaxis=dict(range=[y_min, 100])
    )

    return fig

def create_ttl_with_workingset_graph(ttl_results: Dict[int, float], max_working_set: Dict[int, int]) -> go.Figure:
    """Create graph showing cache hit rate vs TTL with working set size on secondary axis"""

    ttls = sorted(ttl_results.keys())
    hit_rates = [ttl_results[ttl] * 100 for ttl in ttls]
    working_sets = [max_working_set[ttl] for ttl in ttls]

    # Convert seconds to readable labels
    labels = []
    for ttl in ttls:
        if ttl < 60:
            labels.append(f"{ttl}s")
        elif ttl < 3600:
            labels.append(f"{ttl//60}m")
        elif ttl < 86400:
            labels.append(f"{ttl//3600}h")
        else:
            labels.append(f"{ttl//86400}d")

    # Calculate dynamic y-axis minimum for hit rate (round down to nearest 10%)
    min_hit_rate = min(hit_rates)
    y_min = int(min_hit_rate / 10) * 10  # Round down to nearest 10%

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add cache hit rate line (left axis)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=hit_rates,
            mode='lines+markers',
            marker=dict(size=8, color='darkblue'),
            line=dict(width=2, color='darkblue'),
            name='Cache Hit Rate'
        ),
        secondary_y=False
    )

    # Add working set line (right axis)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=working_sets,
            mode='lines+markers',
            marker=dict(size=8, color='darkgreen'),
            line=dict(width=2, color='darkgreen', dash='dash'),
            name='Working Set'
        ),
        secondary_y=True
    )

    # Update axes
    fig.update_xaxes(title_text="TTL")
    fig.update_yaxes(title_text="Cache Hit Rate (%)", secondary_y=False, range=[y_min, 100])
    fig.update_yaxes(title_text="Working Set (Tokens in cache)", secondary_y=True)

    fig.update_layout(
        title='Cache Hit Rate & Working Set vs TTL',
        height=500,
        width=1000,
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

def create_ttl_churnrate_cachehit_graph(ttl_results: Dict[int, float], churn_rates: Dict[int, float]) -> go.Figure:
    """Create graph showing cache hit rate vs TTL with churn rate on secondary axis"""

    ttls = sorted(ttl_results.keys())
    hit_rates = [ttl_results[ttl] * 100 for ttl in ttls]
    churn_values = [churn_rates[ttl] for ttl in ttls]

    # Convert seconds to readable labels
    labels = []
    for ttl in ttls:
        if ttl < 60:
            labels.append(f"{ttl}s")
        elif ttl < 3600:
            labels.append(f"{ttl//60}m")
        elif ttl < 86400:
            labels.append(f"{ttl//3600}h")
        else:
            labels.append(f"{ttl//86400}d")

    # Calculate dynamic y-axis minimum for hit rate (round down to nearest 10%)
    min_hit_rate = min(hit_rates)
    y_min = int(min_hit_rate / 10) * 10

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add cache hit rate line (left axis)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=hit_rates,
            mode='lines+markers',
            marker=dict(size=8, color='darkblue'),
            line=dict(width=2, color='darkblue'),
            name='Cache Hit Rate'
        ),
        secondary_y=False
    )

    # Add churn rate line (right axis)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=churn_values,
            mode='lines+markers',
            marker=dict(size=8, color='darkred'),
            line=dict(width=2, color='darkred', dash='dash'),
            name='Prefill Count'
        ),
        secondary_y=True
    )

    # Add goal line at churn = 1.0
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[1.0] * len(labels),
            mode='lines',
            line=dict(width=1, color='green', dash='dot'),
            name='Optimal Churn (1.0)',
            showlegend=True
        ),
        secondary_y=True
    )

    # Update axes
    fig.update_xaxes(title_text="TTL")
    fig.update_yaxes(title_text="Cache Hit Rate (%)", secondary_y=False, range=[y_min, 100])
    fig.update_yaxes(title_text="Avg Prefill Count", secondary_y=True)

    fig.update_layout(
        title='Cache Hit Rate & Prefill Count vs TTL',
        height=500,
        width=1000,
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02  # Place legend outside plot area on the right
        )
    )

    return fig

def create_ttl_churnrate_workingset_graph(churn_rates: Dict[int, float], max_working_set: Dict[int, int]) -> go.Figure:
    """Create graph showing churn rate vs TTL with working set on secondary axis"""

    ttls = sorted(churn_rates.keys())
    churn_values = [churn_rates[ttl] for ttl in ttls]
    working_sets = [max_working_set[ttl] for ttl in ttls]

    # Convert seconds to readable labels
    labels = []
    for ttl in ttls:
        if ttl < 60:
            labels.append(f"{ttl}s")
        elif ttl < 3600:
            labels.append(f"{ttl//60}m")
        elif ttl < 86400:
            labels.append(f"{ttl//3600}h")
        else:
            labels.append(f"{ttl//86400}d")

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add churn rate line (left axis)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=churn_values,
            mode='lines+markers',
            marker=dict(size=8, color='darkred'),
            line=dict(width=2, color='darkred'),
            name='Prefill Count'
        ),
        secondary_y=False
    )

    # Add goal line at churn = 1.0
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=[1.0] * len(labels),
            mode='lines',
            line=dict(width=1, color='green', dash='dot'),
            name='Optimal Churn (1.0)',
            showlegend=True
        ),
        secondary_y=False
    )

    # Add working set line (right axis)
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=working_sets,
            mode='lines+markers',
            marker=dict(size=8, color='darkgreen'),
            line=dict(width=2, color='darkgreen', dash='dash'),
            name='Working Set'
        ),
        secondary_y=True
    )

    # Update axes
    fig.update_xaxes(title_text="TTL")
    fig.update_yaxes(title_text="Avg Prefill Count", secondary_y=False)
    fig.update_yaxes(title_text="Working Set (Tokens in cache)", secondary_y=True)

    fig.update_layout(
        title='Prefill Count & Working Set vs TTL',
        height=500,
        width=1000,
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02  # Place legend outside plot area on the right
        )
    )

    return fig

def create_working_set_graph(working_set_timeseries: Dict[int, List[Tuple[datetime, int]]]) -> go.Figure:
    """Create graph showing working set size over time for different TTLs"""

    # Select TTLs to plot: 1min, 5min, 15min, 1hour, 1day
    selected_ttls = [60, 300, 900, 3600, 86400]
    ttl_labels = {
        60: '1 minute',
        300: '5 minutes',
        900: '15 minutes',
        3600: '1 hour',
        86400: '1 day'
    }

    colors = {
        60: 'red',
        300: 'blue',
        900: 'green',
        3600: 'purple',
        86400: 'orange'
    }

    fig = go.Figure()

    for ttl in selected_ttls:
        if ttl in working_set_timeseries:
            timeseries = working_set_timeseries[ttl]
            if timeseries:
                timestamps = [ts for ts, _ in timeseries]
                tokens = [tok for _, tok in timeseries]

                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=tokens,
                    mode='lines',
                    name=ttl_labels[ttl],
                    line=dict(width=2, color=colors[ttl])
                ))

    fig.update_layout(
        title='Working Set Size Over Time',
        xaxis_title='Time',
        yaxis_title='Tokens in Cache',
        height=600,
        width=1400,
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

def create_working_set_with_hit_rate_graph(working_set_timeseries: Dict[int, List[Tuple[datetime, int]]],
                                           hit_rate_timeseries: Dict[int, List[Tuple[datetime, float]]]) -> go.Figure:
    """Create dual-axis graph showing working set size + rolling cache hit rate"""

    # Select TTLs to plot: 1min, 5min, 15min, 1hour, 1day
    selected_ttls = [60, 300, 900, 3600, 86400]
    ttl_labels = {
        60: '1 minute',
        300: '5 minutes',
        900: '15 minutes',
        3600: '1 hour',
        86400: '1 day'
    }

    colors = {
        60: 'red',
        300: 'blue',
        900: 'green',
        3600: 'purple',
        86400: 'orange'
    }

    # Calculate dynamic y-axis minimum for hit rate (round down to nearest 10%)
    # Skip first 10 data points to ignore rolling window warmup period
    all_hit_rates = []
    for ttl in selected_ttls:
        if ttl in hit_rate_timeseries:
            hit_timeseries = hit_rate_timeseries[ttl]
            if hit_timeseries and len(hit_timeseries) > 10:
                # Skip first 10 points (warmup period)
                all_hit_rates.extend([rate for _, rate in hit_timeseries[10:]])
            elif hit_timeseries:
                # If fewer than 10 points, use all
                all_hit_rates.extend([rate for _, rate in hit_timeseries])

    if all_hit_rates:
        min_hit_rate = min(all_hit_rates)
        y_min = int(min_hit_rate / 25) * 25  # Round down to nearest 25%
    else:
        y_min = 0  # Fallback if no data

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add working set lines (left axis)
    for ttl in selected_ttls:
        if ttl in working_set_timeseries:
            timeseries = working_set_timeseries[ttl]
            if timeseries:
                timestamps = [ts for ts, _ in timeseries]
                tokens = [tok for _, tok in timeseries]

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=tokens,
                        mode='lines',
                        name=f'{ttl_labels[ttl]} (Working Set)',
                        line=dict(width=2, color=colors[ttl], dash='solid'),
                        yaxis='y'
                    ),
                    secondary_y=False
                )

    # Add cache hit rate lines (right axis)
    for ttl in selected_ttls:
        if ttl in hit_rate_timeseries:
            hit_timeseries = hit_rate_timeseries[ttl]
            if hit_timeseries:
                timestamps = [ts for ts, _ in hit_timeseries]
                hit_rates = [rate for _, rate in hit_timeseries]

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=hit_rates,
                        mode='lines',
                        name=f'{ttl_labels[ttl]} (Cache Hit Rate)',
                        line=dict(width=1, color=colors[ttl], dash='dash'),
                        yaxis='y2'
                    ),
                    secondary_y=True
                )

    # Update axes
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Working Set (Tokens in cache)", secondary_y=False)
    fig.update_yaxes(title_text="Cache Hit Rate (%)", secondary_y=True, range=[y_min, 100])

    fig.update_layout(
        title='Working Set Size & Rolling Cache Hit Rate Over Time',
        height=700,
        width=1600,
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02  # Place legend outside plot area on the right
        )
    )

    return fig

def extract_working_directory(body_str: str) -> Optional[str]:
    """Extract working directory from request body"""
    import re
    try:
        body = json.loads(body_str)
        if 'system' in body and isinstance(body['system'], list):
            for block in body['system']:
                if isinstance(block, dict) and 'text' in block:
                    text = block['text']
                    # Look for env section
                    env_match = re.search(r'<env>(.*?)</env>', text, re.DOTALL)
                    if env_match:
                        # Look for working directory
                        wd_match = re.search(r'Working directory:\s*(.+)', env_match.group(1))
                        if wd_match:
                            return wd_match.group(1).strip()
    except:
        pass
    return None

def process_single_conversation(args, conversation_id: str, output_file: str):
    """Process a single conversation or combined mode"""

    conn = sqlite3.connect(args.db)
    c = conn.cursor()

    # Handle "combined" mode - all requests together
    if conversation_id == 'combined':
        print(f"  Streaming ALL requests (combined mode)...")
        working_directory = "ALL CONVERSATIONS (combined)"

        # Stream processing - don't use fetchall() to avoid loading all into memory
        cursor = c.execute("""
            SELECT id, timestamp, body, response
            FROM requests
            WHERE response IS NOT NULL
            ORDER BY timestamp
        """)

        sim = CompleteCacheSimulator(args.block_size, args.cache_ttl)
        index = 0

        while True:
            row = cursor.fetchone()
            if row is None:
                break

            req_id, ts_str, body_str, resp_str = row
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            body = json.loads(body_str)
            req_type = classify_request(body_str)
            msg_count = len(body.get('messages', []))

            sim.analyze(index, req_id, ts, body_str, resp_str, req_type, msg_count)
            index += 1

            if index % 1000 == 0:
                print(f"    Processed {index:,} requests...", flush=True)

        print(f"    Processed {index:,} requests total")

        if index == 0:
            print(f"  SKIP: No requests found")
            conn.close()
            return None

        # Skip the normal conversation_reqs processing
        conversation_reqs = None

    else:
        # Get indexed requests for time range
        indexed_reqs = c.execute("""
            SELECT r.id, r.timestamp, r.body
            FROM requests r
            INNER JOIN conversation_index ci ON r.id = ci.request_id
            WHERE ci.conversation_id = ?
            ORDER BY r.timestamp
        """, (conversation_id,)).fetchall()

        if not indexed_reqs:
            print(f"  SKIP: No requests found")
            conn.close()
            return None

        # Extract working directory from first request
        working_directory = extract_working_directory(indexed_reqs[0][2])

        indexed_ids = {r[0] for r in indexed_reqs}

        # Get time range with buffer
        first_ts = datetime.fromisoformat(indexed_reqs[0][1].replace('Z', '+00:00'))
        last_ts = datetime.fromisoformat(indexed_reqs[-1][1].replace('Z', '+00:00'))
        range_start = (first_ts - timedelta(seconds=300)).isoformat()
        range_end = (last_ts + timedelta(seconds=300)).isoformat()

        # Get ALL requests
        all_reqs = c.execute("""
            SELECT id, timestamp, body, response
            FROM requests
            WHERE response IS NOT NULL
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp
        """, (range_start, range_end)).fetchall()

        # Filter to conversation using proximity
        def extract_hashes(body_str: str) -> Tuple[str, str, str, int]:
            body = json.loads(body_str)
            messages = body.get('messages', [])
            tools = body.get('tools', [])
            system = body.get('system', [])

            msg_hash = hashlib.md5(json.dumps(messages, separators=(',', ':')).encode()).hexdigest()[:16]
            tool_hash = hashlib.md5(json.dumps(tools, separators=(',', ':')).encode()).hexdigest()[:16]
            sys_hash = hashlib.md5(json.dumps(system, separators=(',', ':')).encode()).hexdigest()[:16]

            return msg_hash, tool_hash, sys_hash, len(messages)

        indexed_info = []
        for req_id, ts_str, body_str in indexed_reqs:
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

        # Run main simulation (for non-combined mode)
        print(f"  Processing {len(conversation_reqs)} requests...", end='', flush=True)
        sim = CompleteCacheSimulator(args.block_size, args.cache_ttl)

        for i, (req_id, ts_str, body_str, resp_str) in enumerate(conversation_reqs):
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            body = json.loads(body_str)
            req_type = classify_request(body_str)
            msg_count = len(body.get('messages', []))

            sim.analyze(i, req_id, ts, body_str, resp_str, req_type, msg_count)

        print(" done")

    # Calculate statistics
    all_with_usage = [a for a in sim.analyses
                      if a.api_cache_read is not None or a.api_cache_creation is not None or a.api_input_tokens is not None]

    # Time statistics
    time_stats = calculate_time_stats(sim.analyses)

    # Context statistics
    context_stats = calculate_context_stats(sim.analyses)

    # Model statistics
    model_stats = calculate_model_stats(sim.analyses)

    # TTL analysis
    ttl_results, max_working_set, working_set_timeseries, hit_rate_timeseries, churn_rates, churn_efficiency = analyze_ttl_impact_fast(
        sim.cached_requests,
        args.block_size,
        interpolation_interval=args.interpolation_interval,
        skip_long_ttls=args.skip_long_ttls,
        show_progress=True
    )
    print()  # New line after progress

    # Create visualizations
    # Simple prompt visualizations
    fig_simple = create_cache_visualization(sim.analyses)
    simple_file = output_file.replace('.html', '_simple_prompt.html')
    fig_simple.write_html(simple_file)

    fig_simple_skinny = create_cache_visualization_skinny(sim.analyses)
    simple_skinny_file = output_file.replace('.html', '_simple_prompt_skinny.html')
    fig_simple_skinny.write_html(simple_skinny_file)

    # Detailed prompt visualizations (message breakdown)
    fig_detailed = create_detailed_cache_visualization(sim.analyses)
    detailed_file = output_file.replace('.html', '_detailed_prompt.html')
    fig_detailed.write_html(detailed_file)

    fig_detailed_skinny = create_detailed_cache_visualization_skinny(sim.analyses)
    detailed_skinny_file = output_file.replace('.html', '_detailed_prompt_skinny.html')
    fig_detailed_skinny.write_html(detailed_skinny_file)

    fig_ttl = create_ttl_graph(ttl_results)
    ttl_file = output_file.replace('.html', '_ttl.html')
    fig_ttl.write_html(ttl_file)

    # Create TTL graph with working set (always generated)
    fig_ttl_workingset = create_ttl_with_workingset_graph(ttl_results, max_working_set)
    ttl_workingset_file = output_file.replace('.html', '_ttl_workingset.html')
    fig_ttl_workingset.write_html(ttl_workingset_file)

    # Create churn rate graphs
    fig_ttl_churnrate_cachehit = create_ttl_churnrate_cachehit_graph(ttl_results, churn_rates)
    ttl_churnrate_cachehit_file = output_file.replace('.html', '_ttl_churnrate_cachehit.html')
    fig_ttl_churnrate_cachehit.write_html(ttl_churnrate_cachehit_file)

    fig_ttl_churnrate_workingset = create_ttl_churnrate_workingset_graph(churn_rates, max_working_set)
    ttl_churnrate_workingset_file = output_file.replace('.html', '_ttl_churnrate_workingset.html')
    fig_ttl_churnrate_workingset.write_html(ttl_churnrate_workingset_file)

    fig_working_set = create_working_set_graph(working_set_timeseries)
    working_set_file = output_file.replace('.html', '_working_set.html')
    fig_working_set.write_html(working_set_file)

    # NEW: Combined working set + hit rate graph
    fig_combined = create_working_set_with_hit_rate_graph(working_set_timeseries, hit_rate_timeseries)
    combined_file = output_file.replace('.html', '_combined.html')
    fig_combined.write_html(combined_file)

    # Write statistics file
    stats_file = output_file.replace('.html', '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("="*80 + "\n")
        if conversation_id == 'combined':
            f.write("COMBINED ANALYSIS - ALL REQUESTS\n")
        else:
            f.write("CONVERSATION STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Conversation ID: {conversation_id}\n")
        if working_directory and conversation_id != 'combined':
            f.write(f"Working Directory: {working_directory}\n")
        f.write(f"Total Requests: {len(sim.analyses)}\n")
        f.write(f"  Streaming: {sum(1 for a in sim.analyses if a.request_type == 'streaming')}\n")
        f.write(f"  Non-streaming: {sum(1 for a in sim.analyses if a.request_type == 'non_streaming')}\n")
        f.write("\n")

        if all_with_usage:
            sim_total = sum(a.total_tokens for a in all_with_usage)
            sim_read = sum(a.cache_read_tokens for a in all_with_usage)
            sim_create = sum(a.cache_creation_tokens for a in all_with_usage)
            sim_input = sim_total - sim_read - sim_create

            api_read = sum(a.api_cache_read or 0 for a in all_with_usage)
            api_create = sum(a.api_cache_creation or 0 for a in all_with_usage)
            api_input = sum(a.api_input_tokens or 0 for a in all_with_usage)
            api_total = api_read + api_create + api_input

            f.write("INPUT TOKENS (5-minute TTL simulation):\n")
            f.write(f"  Simulated Total:    {sim_total:,}\n")
            f.write(f"    Cache Read:       {sim_read:,} ({sim_read/sim_total*100:.1f}%)\n")
            f.write(f"    Cache Creation:   {sim_create:,} ({sim_create/sim_total*100:.1f}%)\n")
            f.write(f"    Input (new):      {sim_input:,} ({sim_input/sim_total*100:.1f}%)\n")
            f.write("\n")
            f.write(f"  API Actual Total:   {api_total:,}\n")
            f.write(f"    Cache Read:       {api_read:,} ({api_read/api_total*100:.1f}%)\n")
            f.write(f"    Cache Creation:   {api_create:,} ({api_create/api_total*100:.1f}%)\n")
            f.write(f"    Input (new):      {api_input:,} ({api_input/api_total*100:.1f}%)\n")
            f.write("\n")
            f.write(f"  Comparison:\n")
            f.write(f"    Total Accuracy:   {sim_total/api_total*100:.1f}%\n")
            if api_read > 0:
                f.write(f"    Cache Read Accuracy: {sim_read/api_read*100:.1f}%\n")

        output_total = sum(a.output_tokens for a in sim.analyses)
        f.write("\n")
        f.write("OUTPUT TOKENS:\n")
        f.write(f"  Total: {output_total:,}\n")
        if all_with_usage and sim_total > 0:
            input_output_ratio = sim_total / output_total if output_total > 0 else 0
            f.write(f"  Input/Output Ratio: {input_output_ratio:.1f}:1\n")

        if model_stats:
            f.write("\n")
            f.write("MODEL USAGE:\n")
            # Sort by input tokens descending
            sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['input_tokens'], reverse=True)
            for model, stats in sorted_models:
                f.write(f"  {model}:\n")
                f.write(f"    Requests: {stats['requests']:,}\n")
                f.write(f"    Input tokens: {stats['input_tokens']:,}\n")
                f.write(f"    Output tokens: {stats['output_tokens']:,}\n")

        if context_stats:
            f.write("\n")
            f.write("CONTEXT LENGTH (tokens per request):\n")
            f.write(f"  Mean: {context_stats['mean']:,.0f}\n")
            f.write(f"  25th percentile: {context_stats['p25']:,}\n")
            f.write(f"  50th percentile: {context_stats['p50']:,} (median)\n")
            f.write(f"  75th percentile: {context_stats['p75']:,}\n")
            f.write(f"  90th percentile: {context_stats['p90']:,}\n")
            f.write(f"  Maximum: {context_stats['max']:,}\n")
            f.write("\n")
            f.write("  Incremental Growth (tokens added per turn):\n")
            f.write(f"    Mean: {context_stats['incremental']['mean']:,.0f}\n")
            f.write(f"    25th percentile: {context_stats['incremental']['p25']:,}\n")
            f.write(f"    50th percentile: {context_stats['incremental']['p50']:,} (median)\n")
            f.write(f"    75th percentile: {context_stats['incremental']['p75']:,}\n")
            f.write(f"    90th percentile: {context_stats['incremental']['p90']:,}\n")
            f.write(f"    Maximum: {context_stats['incremental']['max']:,}\n")

        if time_stats:
            f.write("\n")
            f.write("TIME BETWEEN REQUESTS:\n")
            f.write(f"  Mean: {time_stats['mean']:.1f} seconds\n")
            f.write(f"  25th percentile: {time_stats['p25']:.1f} seconds\n")
            f.write(f"  50th percentile: {time_stats['p50']:.1f} seconds (median)\n")
            f.write(f"  75th percentile: {time_stats['p75']:.1f} seconds\n")
            f.write(f"  90th percentile: {time_stats['p90']:.1f} seconds\n")

        f.write("\n")
        f.write("TTL ANALYSIS (cache hit rates, working set, & prefill count):\n")
        f.write(f"  1 minute:   {ttl_results[60]*100:.1f}% hit, {max_working_set[60]:,} tokens, {churn_rates[60]:.2f} prefill ({churn_efficiency[60]:.1f}% stable)\n")
        f.write(f"  2 minutes:  {ttl_results[120]*100:.1f}% hit, {max_working_set[120]:,} tokens, {churn_rates[120]:.2f} prefill ({churn_efficiency[120]:.1f}% stable)\n")
        f.write(f"  3 minutes:  {ttl_results[180]*100:.1f}% hit, {max_working_set[180]:,} tokens, {churn_rates[180]:.2f} prefill ({churn_efficiency[180]:.1f}% stable)\n")
        f.write(f"  4 minutes:  {ttl_results[240]*100:.1f}% hit, {max_working_set[240]:,} tokens, {churn_rates[240]:.2f} prefill ({churn_efficiency[240]:.1f}% stable)\n")
        f.write(f"  5 minutes:  {ttl_results[300]*100:.1f}% hit, {max_working_set[300]:,} tokens, {churn_rates[300]:.2f} prefill ({churn_efficiency[300]:.1f}% stable)\n")
        f.write(f"  10 minutes: {ttl_results[600]*100:.1f}% hit, {max_working_set[600]:,} tokens, {churn_rates[600]:.2f} prefill ({churn_efficiency[600]:.1f}% stable)\n")
        f.write(f"  15 minutes: {ttl_results[900]*100:.1f}% hit, {max_working_set[900]:,} tokens, {churn_rates[900]:.2f} prefill ({churn_efficiency[900]:.1f}% stable)\n")
        f.write(f"  30 minutes: {ttl_results[1800]*100:.1f}% hit, {max_working_set[1800]:,} tokens, {churn_rates[1800]:.2f} prefill ({churn_efficiency[1800]:.1f}% stable)\n")
        f.write(f"  1 hour:     {ttl_results[3600]*100:.1f}% hit, {max_working_set[3600]:,} tokens, {churn_rates[3600]:.2f} prefill ({churn_efficiency[3600]:.1f}% stable)\n")
        f.write(f"  2 hours:    {ttl_results[7200]*100:.1f}% hit, {max_working_set[7200]:,} tokens, {churn_rates[7200]:.2f} prefill ({churn_efficiency[7200]:.1f}% stable)\n")
        f.write(f"  4 hours:    {ttl_results[14400]*100:.1f}% hit, {max_working_set[14400]:,} tokens, {churn_rates[14400]:.2f} prefill ({churn_efficiency[14400]:.1f}% stable)\n")
        f.write(f"  8 hours:    {ttl_results[28800]*100:.1f}% hit, {max_working_set[28800]:,} tokens, {churn_rates[28800]:.2f} prefill ({churn_efficiency[28800]:.1f}% stable)\n")
        f.write(f"  24 hours:   {ttl_results[86400]*100:.1f}% hit, {max_working_set[86400]:,} tokens, {churn_rates[86400]:.2f} prefill ({churn_efficiency[86400]:.1f}% stable)\n")
        if 259200 in ttl_results:  # Only if 72h was analyzed
            f.write(f"  72 hours:   {ttl_results[259200]*100:.1f}% hit, {max_working_set[259200]:,} tokens, {churn_rates[259200]:.2f} prefill ({churn_efficiency[259200]:.1f}% stable)\n")

        f.write("\n")
        f.write("="*80 + "\n")

    conn.close()

    # Return summary for batch mode
    if all_with_usage and sim_total > 0:
        cache_hit_rate = sim_read / sim_total
        input_output_ratio = sim_total / output_total if output_total > 0 else 0
    else:
        cache_hit_rate = 0
        sim_total = sum(a.total_tokens for a in sim.analyses)
        input_output_ratio = sim_total / output_total if output_total > 0 else 0

    return {
        'requests': len(sim.analyses),
        'working_directory': working_directory,
        'input_tokens': sim_total,
        'output_tokens': output_total,
        'input_output_ratio': input_output_ratio,
        'cache_hit_rate': cache_hit_rate,
        'ttl_1min': ttl_results[60],
        'ttl_5min': ttl_results[300],
        'ttl_30min': ttl_results[1800],
        'ttl_1hour': ttl_results[3600],
        'ttl_24hour': ttl_results[86400],
        'working_set_1min': max_working_set[60],
        'working_set_5min': max_working_set[300],
        'working_set_15min': max_working_set[900],
        'working_set_1hour': max_working_set[3600],
        'working_set_24hour': max_working_set[86400],
        'files': {
            'cache': output_file,
            'ttl': ttl_file,
            'working_set': working_set_file,
            'combined': combined_file,
            'stats': stats_file
        }
    }

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Complete cache visualization')
    parser.add_argument('db', help='Path to requests.db')
    parser.add_argument('--conversation-id', required=True, help='Conversation ID, "all", or "combined"')
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--cache-ttl', type=int, default=300)
    parser.add_argument('--output', default='viz.html')
    parser.add_argument('--output-dir', help='Output directory for batch mode')
    parser.add_argument('--newest', action='store_true', help='Process newest conversations first (for --conversation-id all)')
    parser.add_argument('--skip-long-ttls', action='store_true', help='Skip 72h TTL analysis')
    parser.add_argument('--interpolation-interval', type=int, default=15, help='Minutes between interpolation points (default: 15)')
    args = parser.parse_args()

    if args.conversation_id == 'combined':
        # Combined mode - all requests as one conversation
        print("="*80)
        print("COMBINED MODE - ALL REQUESTS")
        print("="*80)
        print()

        output_file = args.output if args.output != 'viz.html' else 'combined.html'
        result = process_single_conversation(args, 'combined', output_file)

        if result:
            print()
            print("="*80)
            print("COMBINED ANALYSIS COMPLETE")
            print("="*80)
            print(f"Total requests: {result['requests']}")
            print(f"Input tokens: {result['input_tokens']:,}")
            print(f"Output tokens: {result['output_tokens']:,}")
            print(f"Input/Output ratio: {result['input_output_ratio']:.1f}:1")
            print(f"Cache hit rate (5-min TTL): {result['cache_hit_rate']*100:.1f}%")
            print()
            print(f"Working set sizes:")
            print(f"  1 min:  {result['working_set_1min']:,} tokens")
            print(f"  5 min:  {result['working_set_5min']:,} tokens")
            print(f"  15 min: {result['working_set_15min']:,} tokens")
            print(f"  1 hour: {result['working_set_1hour']:,} tokens")
            print(f"  24 hr:  {result['working_set_24hour']:,} tokens")
            print()
            print(f"Files generated:")
            print(f"  - {result['files']['cache']}")
            print(f"  - {result['files']['ttl']}")
            print(f"  - {result['files']['working_set']}")
            print(f"  - {result['files']['combined']}")
            print(f"  - {result['files']['stats']}")

    elif args.conversation_id == 'all':
        # Batch mode
        output_dir = args.output_dir or 'visualizations'
        os.makedirs(output_dir, exist_ok=True)

        conn = sqlite3.connect(args.db)

        # Choose query based on --newest flag
        if args.newest:
            conversations = conn.execute("""
                SELECT conversation_id, COUNT(*) as count
                FROM conversation_index
                GROUP BY conversation_id
                HAVING count >= 10
                ORDER BY MAX(timestamp) DESC
            """).fetchall()
            sort_mode = "newest first"
        else:
            conversations = conn.execute("""
                SELECT conversation_id, COUNT(*) as count
                FROM conversation_index
                GROUP BY conversation_id
                HAVING count >= 10
                ORDER BY count DESC
            """).fetchall()
            sort_mode = "most requests first"

        conn.close()

        print("="*80)
        print("BATCH PROCESSING")
        print("="*80)
        print(f"Conversations: {len(conversations)}")
        print(f"Sort order: {sort_mode}")
        print(f"Output directory: {output_dir}/")
        print()

        processed_count = 0
        skipped_count = 0
        error_count = 0

        for i, (conv_id, count) in enumerate(conversations):
            print(f"[{i+1}/{len(conversations)}] {conv_id[:16]}... ({count} requests)", end=' ')

            output_file = os.path.join(output_dir, f'{conv_id}.html')

            # Check if all output files already exist
            expected_files = [
                output_file,
                output_file.replace('.html', '_ttl.html'),
                output_file.replace('.html', '_working_set.html'),
                output_file.replace('.html', '_combined.html'),
                output_file.replace('.html', '_stats.txt')
            ]

            if all(os.path.exists(f) for f in expected_files):
                print("⊘ (already exists, skipping)")
                skipped_count += 1
                continue

            try:
                result = process_single_conversation(args, conv_id, output_file)
                if result:
                    print(f"✓")
                    processed_count += 1
                    # Show working directory if available
                    if result.get('working_directory'):
                        wd_basename = os.path.basename(result['working_directory'])
                        print(f"    Dir: {wd_basename}")
                    print(f"    Input: {result['input_tokens']:,} | Output: {result['output_tokens']:,} | I/O Ratio: {result['input_output_ratio']:.1f}:1")
                    print(f"    Cache: 1m={result['ttl_1min']*100:.1f}% 5m={result['ttl_5min']*100:.1f}% 30m={result['ttl_30min']*100:.1f}% 1h={result['ttl_1hour']*100:.1f}% 24h={result['ttl_24hour']*100:.1f}%")
                    print(f"    WorkSet: 1m={result['working_set_1min']:,} 5m={result['working_set_5min']:,} 15m={result['working_set_15min']:,} 1h={result['working_set_1hour']:,} 24h={result['working_set_24hour']:,}")
                else:
                    print()
            except Exception as e:
                print(f" ✗ Error: {e}")
                error_count += 1

        print()
        print(f"✅ Batch processing complete: {output_dir}/")
        print(f"   Processed: {processed_count}, Skipped: {skipped_count}, Errors: {error_count}")

    else:
        # Single conversation mode
        result = process_single_conversation(args, args.conversation_id, args.output)
        if result:
            print()
            print("="*80)
            print("SUMMARY")
            print("="*80)
            if result.get('working_directory'):
                print(f"Working directory: {result['working_directory']}")
            print(f"Requests: {result['requests']}")
            print(f"Input tokens: {result['input_tokens']:,}")
            print(f"Output tokens: {result['output_tokens']:,}")
            print(f"Input/Output ratio: {result['input_output_ratio']:.1f}:1")
            print(f"Cache hit rate (5-min TTL): {result['cache_hit_rate']*100:.1f}%")
            print()
            print(f"Files generated:")
            print(f"  - {result['files']['cache']}")
            print(f"  - {result['files']['ttl']}")
            print(f"  - {result['files']['working_set']}")
            print(f"  - {result['files']['combined']}")
            print(f"  - {result['files']['stats']}")

if __name__ == "__main__":
    main()
