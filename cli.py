"""
Command Line Interface (CLI) for CILI RAG System.
Compatible with Python 3.11+.
"""

import sys
import argparse
from pathlib import Path

from config import default_config, SAMPLE_DATA_DIR
from rag_engine import RAGEngine
import groq_chat

def main():
    parser = argparse.ArgumentParser(description="CILI RAG System - CLI Tool")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and index documents")
    ingest_parser.add_argument("--path", "-p", type=str, help="Path to file or directory to ingest")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question against the index")
    query_parser.add_argument("question", type=str, help="Question to ask")
    query_parser.add_argument("--top-k", "-k", type=int, default=4, help="Top K context chunks")
    query_parser.add_argument("--no-hybrid", action="store_true", help="Disable hybrid search")

    # Info command
    subparsers.add_parser("info", help="View vector store index status")

    # Interactive Groq chat command
    subparsers.add_parser("interactive", help="Start interactive Groq AI chat session with RAG Tool")

    args = parser.parse_args()

    engine = RAGEngine(config=default_config)

    if args.command == "ingest":
        path = Path(args.path) if args.path else SAMPLE_DATA_DIR
        if not path.exists():
            print(f"Error: Path '{path}' does not exist.")
            sys.exit(1)

        print(f"[*] Ingesting documents from: {path} ...")
        if path.is_file():
            count = engine.ingest_files([path])
        else:
            count = engine.ingest_directory(path)

        print(f"[✓] Successfully indexed {count} text chunks!")

    elif args.command == "query":
        if engine.vector_store.count() == 0:
            print("[!] Vector store is empty. Ingesting sample data first...")
            engine.ingest_directory(SAMPLE_DATA_DIR)

        print(f"[*] Querying: '{args.question}'...")
        result = engine.query(
            question=args.question,
            top_k=args.top_k,
            use_hybrid=not args.no_hybrid
        )

        print("\n" + "=" * 60)
        print(f"ANSWER (Provider: {result.llm_provider}):")
        print("=" * 60)
        print(result.answer)
        print("\n" + "=" * 60)
        print(f"RETRIEVED SOURCES ({len(result.sources)} chunks, Avg Confidence: {result.confidence_score}):")
        print("=" * 60)
        for i, src in enumerate(result.sources, 1):
            print(f"\n[{i}] {src.metadata.get('source', 'doc')} | Score: {src.similarity_score}")
            print(f"    Excerpt: {src.text[:150]}...")

    elif args.command == "info":
        count = engine.vector_store.count()
        print("\n=== CILI RAG System Status ===")
        print(f"Vector Database Path : {default_config.vector_db_path}")
        print(f"Total Indexed Chunks : {count}")
        print(f"Embedding Provider   : {engine.embedding_provider.__class__.__name__}")
        print(f"LLM Model Config     : {default_config.groq_model}")
        print("==============================\n")

    elif args.command == "interactive":
        groq_chat.main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
