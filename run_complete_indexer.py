#!/usr/bin/env python
"""
run_complete_indexer.py - Index ALL data with rate limiting
"""

import os
import sys
import time
from pathlib import Path


def main():
    print("COMPLETE DATA INDEXER FOR MEMBER MESSAGES")
    print()
    print("This script will:")
    print("1. Fetch ALL messages from the API (with rate limiting)")
    print("2. Create embeddings for each message")
    print("3. Build a complete FAISS index")
    print("4. Save to data/faiss_index_complete/")
    print()
    print("IMPORTANT:")
    print("- This will take 15-30 minutes due to rate limiting")
    print("- The script fetches 200 messages every 10 seconds")
    print("- Total expected messages: ~3,349")
    print()

    response = input("Do you want to proceed? (y/n): ")
    if response.lower() != "y":
        print("Cancelled.")
        return

    print()
    print("Starting complete indexing process...")

    # Run the indexer
    from src.message_rag.faiss_indexer_complete import main as run_indexer

    start_time = time.time()
    builder = run_indexer()

    if builder:
        elapsed = time.time() - start_time
        print()
        print("INDEXING COMPLETE!")
        print(f"Time taken: {elapsed/60:.1f} minutes")
        print(f"Total messages indexed: {len(builder.messages)}")
        print(f"Index saved to: data/faiss_index_complete/")
        print()
        print("Next steps:")
        print("1. Update your server to use the complete index")
        print("2. Restart the server")
        print("3. Test with complex queries")
    else:
        print("Indexing failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
