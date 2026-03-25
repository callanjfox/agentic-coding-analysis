#!/usr/bin/env python3
"""
Build Message ID Index

Creates a fast-lookup index mapping Anthropic message IDs to database request IDs.
This enables accurate conversation-to-request mapping.
"""

import sqlite3
import json
import sys

def build_message_id_index(db_path='./requests.db'):
    """Build index of message IDs from database"""

    print("Building message ID index...")
    print(f"Database: {db_path}")
    print()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get total count
    cursor.execute("SELECT COUNT(*) FROM requests WHERE response IS NOT NULL")
    total = cursor.fetchone()[0]
    print(f"Total requests with responses: {total:,}")
    print()

    # Extract message IDs
    cursor.execute("SELECT id, response FROM requests WHERE response IS NOT NULL")

    message_id_index = {}  # message_id -> db_request_id
    count = 0

    for req_id, response_json in cursor.fetchall():
        count += 1
        if count % 5000 == 0:
            print(f"  Progress: {count:,}/{total:,} ({count/total*100:.1f}%)")

        try:
            response = json.loads(response_json)

            if 'body' in response:
                resp_body = response['body']
                if isinstance(resp_body, str):
                    resp_body = json.loads(resp_body)

                msg_id = resp_body.get('id')
                if msg_id and msg_id.startswith('msg_'):
                    message_id_index[msg_id] = req_id
        except:
            pass

    conn.close()

    print(f"  Complete!")
    print()
    print(f"Extracted {len(message_id_index):,} unique message IDs")

    # Save to file
    output_file = 'message_id_index.json'
    with open(output_file, 'w') as f:
        json.dump(message_id_index, f)

    print(f"Saved to: {output_file}")
    print()
    print(f"Coverage: {len(message_id_index)/total*100:.1f}% of requests have message IDs")

    return message_id_index

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else './requests.db'
    build_message_id_index(db_path)
