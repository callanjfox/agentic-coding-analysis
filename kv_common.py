#!/usr/bin/env python3
"""
Shared utilities for KV cache analysis tools.

Common functions used across multiple scripts for request classification,
content normalization, tokenization, and cache simulation.
"""

import copy
import json
import hashlib
import tiktoken
from datetime import datetime
from typing import Any, List
from dataclasses import dataclass


# Singleton tokenizer instance
_tokenizer = None


def get_tokenizer():
    """Get or create singleton tiktoken GPT-4 encoder."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model("gpt-4")
    return _tokenizer


def classify_request(body_str: str) -> str:
    """Classify request type based on max_tokens and stream flag.

    Claude Code sends two requests per turn:
    - Streaming: max_tokens=32000, stream=True (for UI responsiveness)
    - Non-streaming: max_tokens~=21333, stream=False (for tool execution)

    Returns: 'streaming', 'non_streaming', or 'unknown'
    """
    body = json.loads(body_str)
    max_tokens = body.get('max_tokens', 0)
    stream = body.get('stream', False)

    if max_tokens == 32000 and stream:
        return 'streaming'
    elif max_tokens in [21333, 21334, 21332]:
        return 'non_streaming'
    else:
        return 'unknown'


def normalize_for_cache(obj: Any) -> Any:
    """Remove cache_control and signature markers recursively.

    These fields vary between requests but don't affect cache content:
    - cache_control: moves to mark the last block as cacheable
    - signature: thinking block signatures (256-20K chars, change each request)
    """
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


def create_chained_hash(token_ids: List[int], prev_hash: str, seq_num: int) -> str:
    """Create a hash that chains with previous block.

    Each block's hash depends on the previous block's hash, ensuring that
    identical content at the same position produces identical hashes (and
    different positions produce different hashes even for same content).
    """
    content = f"{prev_hash}:{seq_num}:" + " ".join(map(str, token_ids))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class CacheChunk:
    """Represents a cacheable token block with TTL tracking."""
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


def compact_json(obj) -> str:
    """Format JSON with compact arrays on single lines.

    Produces human-readable JSON where primitive arrays (ints, floats, strings)
    are kept on a single line rather than one element per line.
    """
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
