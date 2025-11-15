"""
src/message_rag/api_server.py - Complete Working FastAPI Server
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime

from .faiss_indexer import FAISSIndexBuilder
from .rag_engine import MessageRAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Member Message QA System",
    description="Natural language Q&A system for member messages",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
index_builder: Optional[FAISSIndexBuilder] = None
rag_engine: Optional[MessageRAGEngine] = None


class QuestionRequest(BaseModel):
    """Request model for questions"""

    question: str = Field(..., description="Natural language question about member data")


class AnswerResponse(BaseModel):
    """Response model - matches the required format"""

    answer: str = Field(..., description="AI-generated answer")


@app.on_event("startup")
async def startup_event():
    """Initialize everything on startup"""
    global index_builder, rag_engine

    try:
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY not set!")
            raise ValueError(
                "OpenAI API key not found. " "Please set: export OPENAI_API_KEY='sk-your-key-here'"
            )

        # Load FAISS index
        logger.info("Loading FAISS index...")
        index_builder = FAISSIndexBuilder()

        if not index_builder.load_index():
            logger.error("FAISS index not found")
            raise RuntimeError(
                "FAISS index not found. Please build it first:\n" "  uv run python run_indexer.py"
            )

        stats = index_builder.get_statistics()
        logger.info(
            f"Loaded {stats['total_messages']} messages from {stats['total_users']} users"
        )

        # Initialize RAG engine
        logger.info("Initializing RAG engine")

        # Use environment variable for model if set, default to gpt-4-turbo
        model_name = os.getenv("OPENAI_MODEL", "gpt-4-turbo")

        rag_engine = MessageRAGEngine(
            index_builder=index_builder,
            model_name=model_name,
            temperature=0.2,  # Keep responses focused
        )

        logger.info(f"RAG engine ready with model: {model_name}")
        logger.info("System ready at http://localhost:8000")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint with system information"""
    if not rag_engine:
        return {"status": "initializing", "message": "System is starting up..."}

    return {
        "service": "Member Message QA System",
        "status": "ready",
        "model": rag_engine.model_name,
        "endpoints": {
            "/ask": "POST - Ask a question (returns {answer: '...'})",
            "/health": "GET - Health check",
            "/stats": "GET - System statistics",
            "/docs": "GET - Interactive API documentation",
        },
        "example_usage": {
            "method": "POST",
            "url": "/ask",
            "body": {"question": "When is Layla planning her trip to London?"},
            "response": {"answer": "Based on the messages..."},
        },
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint - Ask a natural language question about member data

    This endpoint accepts a question and returns an AI-generated answer
    based on the member messages in the system.

    Request:
        {
            "question": "Your question here"
        }

    Response:
        {
            "answer": "The answer based on member messages"
        }

    Example questions:
    - "When is Layla planning her trip to London?"
    - "How many restaurant reservations have been made?"
    - "Which clients prefer luxury accommodations?"
    - "What are the most popular travel destinations?"
    """

    if not rag_engine:
        raise HTTPException(
            status_code=503, detail="System not initialized. Please wait and try again."
        )

    # Validate question
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        logger.info(f"Processing question: {request.question}")

        # Get answer from RAG engine
        result = rag_engine.answer_question(request.question)

        # Log if there was an error but still got an answer
        if "error" in result and result.get("answer"):
            logger.warning(f"Handled error: {result.get('error')}")

        # Return the answer in the required format
        return AnswerResponse(answer=result["answer"])

    except Exception as e:
        logger.error(f"Error processing question: {e}")

        # Return a user-friendly error message
        error_message = (
            "I encountered an error processing your question. "
            "Please try rephrasing or asking a simpler question."
        )

        return AnswerResponse(answer=error_message)


@app.get("/health")
async def health_check():
    """Health check endpoint"""

    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "running",
            "faiss_index": "loaded" if index_builder else "not_loaded",
            "rag_engine": "ready" if rag_engine else "not_initialized",
        },
    }

    # Add model info if available
    if rag_engine:
        health_status["components"]["model"] = rag_engine.model_name
        health_status["components"]["message_count"] = (
            len(index_builder.messages) if index_builder else 0
        )

    return health_status


@app.get("/stats")
async def get_statistics():
    """Get detailed statistics about the system"""

    if not index_builder:
        raise HTTPException(status_code=503, detail="System not initialized")

    stats = index_builder.get_statistics()

    # Add RAG engine info
    if rag_engine:
        stats["rag_engine"] = {
            "model": rag_engine.model_name,
            "status": "ready",
            "vectorstore_size": len(index_builder.messages),
        }

    return stats


@app.get("/examples")
async def get_example_questions():
    """Get example questions to try"""

    return {
        "simple_questions": [
            "When is Layla planning her trip to London?",
            "What restaurants has Lorenzo mentioned?",
            "Which flights has Sophia booked?",
            "What hotels has Hans mentioned?",
        ],
        "counting_questions": [
            "How many restaurant reservations have been made?",
            "How many clients have requested flights?",
            "How many different hotels are mentioned?",
            "How many VIP tickets have been requested?",
        ],
        "analysis_questions": [
            "Which clients prefer luxury accommodations?",
            "What are the most popular travel destinations?",
            "What dining preferences have clients expressed?",
            "What special events are clients planning to attend?",
        ],
        "usage": {
            "curl": 'curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d \'{"question": "YOUR_QUESTION"}\'',
            "python": """
import requests
response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "YOUR_QUESTION"}
)
print(response.json()["answer"])
""",
        },
    }


# Error handlers
@app.exception_handler(404)
async def not_found(request, exc):
    return {
        "error": "Endpoint not found",
        "available_endpoints": ["/ask", "/health", "/stats", "/examples"],
        "documentation": "/docs",
    }


@app.exception_handler(500)
async def internal_error(request, exc):
    return {"error": "Internal server error", "message": "Please try again or contact support"}


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("Member Message QA System")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OpenAI API key not set!")
        print("")
        print("Please set your API key:")
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        print("")
        print("Get your key from: https://platform.openai.com/api-keys")
        exit(1)

    # Check for FAISS index
    if not os.path.exists("data/faiss_index/index.bin"):
        print("❌ ERROR: FAISS index not found!")
        print("")
        print("Please build the index first:")
        print("  uv run python run_indexer.py")
        print("")
        exit(1)

    print("✓ All requirements met")
    print("")
    print("Starting server...")
    print("=" * 60)

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="info")
