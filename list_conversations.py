#!/usr/bin/env python3
"""
List Conversations from requests.db

Shows available conversations with message counts for use with --conversation-id filtering
"""

import sqlite3
import argparse

def main():
    parser = argparse.ArgumentParser(description='List conversations in requests.db')
    parser.add_argument('db', help='Path to requests.db')
    parser.add_argument('--limit', type=int, default=20, help='Number of conversations to show')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # Check if conversation_index exists
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if 'conversation_index' not in tables:
        print("❌ conversation_index table not found in database")
        print()
        print("This database doesn't have conversation tracking.")
        print("Use --start-date filtering instead.")
        return

    # Get conversations
    conversations = conn.execute("""
        SELECT
            conversation_id,
            COUNT(*) as message_count,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message
        FROM conversation_index
        GROUP BY conversation_id
        ORDER BY COUNT(*) DESC
        LIMIT ?
    """, (args.limit,)).fetchall()

    conn.close()

    if not conversations:
        print("❌ No conversations found in conversation_index table")
        return

    print(f"=" * 100)
    print(f"CONVERSATIONS IN DATABASE (Top {len(conversations)})")
    print(f"=" * 100)
    print()

    print(f"{'Conversation ID':<40} {'Messages':<10} {'First Message':<20} {'Last Message'}")
    print("-" * 100)

    for conv in conversations:
        conv_id = conv['conversation_id']
        msg_count = conv['message_count']
        first = conv['first_message'][:19] if conv['first_message'] else 'N/A'
        last = conv['last_message'][:19] if conv['last_message'] else 'N/A'

        print(f"{conv_id:<40} {msg_count:<10} {first:<20} {last}")

    print()
    print(f"=" * 100)
    print("TO USE A CONVERSATION")
    print(f"=" * 100)
    print()
    print("Copy the full conversation ID and use it with --conversation-id:")
    print()
    print(f"  python3 content_structure_cache_simulator.py requests.db \\")
    print(f"    --conversation-id {conversations[0]['conversation_id']}")
    print()
    print("Or visualize it:")
    print()
    print(f"  python3 cache_visualizer.py requests.db \\")
    print(f"    --conversation-id {conversations[0]['conversation_id']} \\")
    print(f"    --output conversation_{conversations[0]['conversation_id'][:8]}.html")

if __name__ == "__main__":
    main()
