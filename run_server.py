#!/usr/bin/env python
"""
run_server.py - Run the complete QA system
"""

import os
import sys


def check_requirements():
    """Check all requirements before starting"""

    print("Checking system requirements...")
    print("-" * 40)

    errors = []

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        errors.append(
            "OpenAI API key not set!\n" "   Fix: export OPENAI_API_KEY='sk-your-key-here'"
        )
    else:
        print(f"✓ OpenAI API key found ({api_key[:10]}...)")

    # Check FAISS index
    index_path = "data/faiss_index/index.bin"
    if not os.path.exists(index_path):
        errors.append("FAISS index not found!\n" "   Fix: uv run python run_indexer.py")
    else:
        print("FAISS index found")

    # Check imports
    # try:
    #     from langchain_openai import ChatOpenAI
    #     from langchain_community.vectorstores import FAISS
    #     from langchain_core.prompts import PromptTemplate
    #     from langchain_core.documents import Document

    #     print("✓ All LangChain imports working")
    # except ImportError as e:
    #     errors.append(
    #         f"LangChain import error: {e}\n"
    #         "   Fix: uv add langchain-openai langchain-community langchain-core"
    #     )

    # Check model setting
    model = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
    print(f"Using model: {model}")

    return errors


def main():
    """Main function to run the server"""

    print("Member Message QA System")
    print()

    # Check requirements
    errors = check_requirements()

    if errors:
        print("\n Setup Issues Found:")
        for error in errors:
            print(error)
        print("\nPlease fix the issues above and try again.")
        sys.exit(1)

    print("\nAll checks passed!")
    print("-" * 40)

    # Show instructions
    print("\n Server starting at: http://localhost:8000")
    print(" API docs at: http://localhost:8000/docs")
    print(" Health check: http://localhost:8000/health")

    print("\n Test the API:")
    print("curl -X POST http://localhost:8000/ask \\")
    print("  -H 'Content-Type: application/json' \\")
    print('  -d \'{"question": "When is Layla planning her trip to London?"}\'')

    print("\nPress Ctrl+C to stop the server")

    # Import and run uvicorn
    try:
        import uvicorn

        uvicorn.run(
            "src.message_rag.api_server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
    except Exception as e:
        print(f"\n Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
