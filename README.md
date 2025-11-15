# Message-RAG: Intelligent Q&A System for Member Messages

A high-performance RAG (Retrieval-Augmented Generation) system built with FAISS, LangChain, and OpenAI GPT-4 for intelligent question-answering over member message data.

## Features

- **Complete Message Indexing**: Fetches and indexes all member messages with intelligent rate limiting
- **FAISS Vector Search**: Lightning-fast similarity search using Facebook's FAISS library
- **RAG Pipeline**: Combines retrieval with GPT-4 for accurate, context-aware answers
- **Docker Deployment**: Production-ready containerization with ngrok tunneling support
- **RESTful API**: Simple HTTP endpoints for easy integration
- **Smart Context Selection**: Automatically retrieves the most relevant messages for each query

## Quick Start

### Prerequisites

- Python 3.13+
- OpenAI API key
- Docker (for deployment)
- ngrok (for public tunneling)

### Installation

1. **Clone the repository**
```bash
git clone git@github.com:SaumyaGupta-99/message-rag.git
cd message-rag
```

2. **Install dependencies using uv**
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

3. **Set your OpenAI API key**
```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

### Building the Index

Before you can query the system, you need to build the FAISS index:

```bash
uv run python src/message_rag/faiss_indexer.py
```

This will:
- Fetch all messages from the API (with automatic rate limiting)
- Create embeddings using `all-MiniLM-L6-v2` model
- Build a FAISS index for fast similarity search
- Save the index to `data/faiss_index/`

**Note**: Initial indexing takes several minutes due to API rate limiting (200 messages per batch, 10 seconds between batches).

### Running the Server

#### Local Development
```bash
uv run python run_server.py
```

The server will start at `http://localhost:8000`

#### Docker Deployment
```bash
# Build the Docker image
docker build -t message-rag:latest .

# Run the container
docker run -d \
    --name message-rag-server \
    -p 8000:8000 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/data:/app/data:ro" \
    message-rag:latest

# Start ngrok tunnel for public access
ngrok http 8000
```

## API Usage

### Ask a Question

**Endpoint**: `POST /ask`

**Request**:
```json
{
  "question": "When is Layla planning her trip to London?"
}
```

**Response**:
```json
{
  "answer": "Layla Kawaguchi is planning her trip to London for next month from the date mentioned in her message on 2025-10-23T19:27:48.166917+00:00, where she requests a chauffeur-driven Bentley for her stay in London. This indicates her planned arrival in London would be in November 2025."
}
```

```json
{"question": "Which restaurants did Amira visit?"}'
```

**Response**:
```json
{
  "answer": "Amina Van Den Berg visited Narisawa, as mentioned in her message dated 2025-02-03. The other messages mention dining experiences in Barcelona, Dubai, and Sydney, but specific restaurant names are not provided in the context."
}
```

**Request**:
```json
{"question": "How many restaurant reservations have been made?"}'
```

**Response**:
```json
{"answer":"Based on the messages provided, here are the instances where restaurant reservations are mentioned:\n\n1. Fatima El-Tahir's message about confirming a reservation for Valentine's Day.\n2. Sophia Al-Farsi's request for last-minute reservations for ten people.\n3. Layla Kawaguchi's mention of her preferred restaurant being fully booked and seeking alternatives (implies a need for a reservation).\n4. Hans Müller's appreciation for the Michelin restaurant reservations.\n5. Vikram Desai's request to reconfirm a restaurant reservation at 8 pm on Saturday.\n6. Hans Müller's request for a reservation at a top-rated Parisian restaurant.\n\nEach of these messages indicates a unique occurrence of a restaurant reservation or a request related to one. Therefore, there are six restaurant reservations mentioned.\n\n**Total: 6 restaurant reservations**"}%
```

**Request**:
```json
{"question": "What questions did Hans have?"}
```

**Response**:
```json
{
  "answer": "Hans Müller asked the following questions:\n\n1. On 2025-05-19, he inquired about the arrangements for his upcoming trip to Tokyo.\n2. On 2025-09-21, he requested the contact details of an artist from a gallery event he attended the previous day.\n3. On 2025-05-05, he expressed interest in booking a Scandinavian cruise and asked for dates and options.\n4. On 2024-11-19, he asked if his reservation on the Orient Express had been confirmed.\n5. On 2025-04-28, he requested hypoallergenic pillows in his hotel suite.\n6. On 2025-10-13, he asked for a chauffeur to drive him around Amsterdam during that week.\n7. On 2025-09-29, he inquired if arrangements had been made for car service during his trip to New York."
}
```

### Other Endpoints

- `GET /` - System information and usage examples
- `GET /health` - Health check endpoint
- `GET /stats` - Detailed statistics about indexed messages
- `GET /examples` - Example questions to try
- `GET /docs` - Interactive API documentation (Swagger UI)

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI Server                       │
│                    (src/message_rag/api_server.py)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      RAG Engine                              │
│                 (src/message_rag/rag_engine.py)             │
│  • Question Processing                                       │
│  • Context Retrieval                                        │
│  • GPT-4 Integration                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    FAISS Index Builder                       │
│              (src/message_rag/faiss_indexer.py)             │
│  • Message Fetching                                         │
│  • Embedding Generation                                     │
│  • Index Management                                         │
└──────────────────────────────────────────────────────────────┘
```

### Key Technologies

- **FAISS**: Facebook AI Similarity Search for efficient vector storage and retrieval
- **LangChain**: Framework for building LLM applications with retrieval
- **Sentence Transformers**: Creates embeddings using `all-MiniLM-L6-v2` model
- **OpenAI GPT-4**: Large language model for generating intelligent answers
- **FastAPI**: Modern, fast web framework for building APIs
- **Docker**: Containerization for consistent deployment
- **ngrok**: Secure tunneling for public access

## Project Structure

```
message-rag/
├── src/
│   └── message_rag/
│       ├── __init__.py
│       ├── faiss_indexer.py    # FAISS index building and management
│       ├── rag_engine.py       # RAG pipeline implementation
│       └── api_server.py       # FastAPI server
├── data/
│   └── faiss_index/            # Stored FAISS index files
│       ├── index.bin           # FAISS index
│       ├── metadata.pkl        # Message metadata
│       └── statistics.json     # Index statistics
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── pyproject.toml              # Project dependencies
├── run_server.py               # Server startup script
└── README.md                   # This file
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: `gpt-4-turbo`)
- `PYTHONPATH`: Set to `/app` in Docker

### Index Configuration

The indexer can be configured in `faiss_indexer.py`:

```python
# Batch size for API fetching
batch_size = 200

# Delay between API calls (rate limiting)
delay_seconds = 10

# Embedding model
model_name = "all-MiniLM-L6-v2"

# FAISS index type (automatic based on size)
# < 10,000 messages: Flat index
# > 10,000 messages: IVF index
```

## Docker Deployment

### Building the Image

```bash
docker build -t message-rag:latest .
```

### Running with Docker Compose

```bash
docker-compose up -d
```

### Manual Docker Run

```bash
docker run -d \
    --name message-rag-server \
    -p 8000:8000 \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -v "$(pwd)/data:/app/data:ro" \
    message-rag:latest
```

## Development

### Running Tests

```bash
uv run pytest tests/
```

### Code Formatting

```bash
uv run black src/
uv run ruff check src/
```

### Rebuilding the Index

```bash
# Remove old index
rm -rf data/faiss_index/

# Rebuild
uv run python src/message_rag/faiss_indexer.py
```

## Potential Updates

This section describes enhancements that would have been added with more time.  
They focus on improving efficiency, quality, and usability.

### Smarter chunking
A more advanced chunking process would be introduced using overlapping context windows.  
This would allow the system to capture shared information across segments and improve coverage of diverse topics.

### Chain of Thought prompting for clearer reasoning
The prompting strategy would be extended with structured reasoning steps to make explanations easier to follow.  
This would strengthen interpretability and help users understand how answers are produced.

### Automatic citation generation
A mechanism would be added to attach references to each answer using the retrieved source passages.  
This would produce more holistic and verifiable responses.

### User interface improvements
A simple interface would be added to support better interaction outside the notebook environment.  
This may include input fields for queries, highlighted retrieved evidence, and a layout that shows reasoning and citations in an organized way.

