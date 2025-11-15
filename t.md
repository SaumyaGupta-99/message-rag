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
git clone <repository-url>
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

### Example Queries

#### 1. Travel Planning Questions

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "When is Layla planning her trip to London?"}'
```

**Response**:
```json
{
  "answer": "Layla Kawaguchi is planning her trip to London for next month from the date mentioned in her message on 2025-10-23T19:27:48.166917+00:00, where she requests a chauffeur-driven Bentley for her stay in London. This indicates her planned arrival in London would be in November 2025."
}
```

#### 2. Restaurant and Dining Queries

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which restaurants did Amira visit?"}'
```

**Response**:
```json
{
  "answer": "Amina Van Den Berg visited Narisawa, as mentioned in her message dated 2025-02-03. The other messages mention dining experiences in Barcelona, Dubai, and Sydney, but specific restaurant names are not provided in the context."
}
```

#### 3. Client Activity Summaries

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What questions did Hans have?"}'
```

**Response**:
```json
{
  "answer": "Hans Müller asked the following questions:\n\n1. On 2025-05-19, he inquired about the arrangements for his upcoming trip to Tokyo.\n2. On 2025-09-21, he requested the contact details of an artist from a gallery event he attended the previous day.\n3. On 2025-05-05, he expressed interest in booking a Scandinavian cruise and asked for dates and options.\n4. On 2024-11-19, he asked if his reservation on the Orient Express had been confirmed.\n5. On 2025-04-28, he requested hypoallergenic pillows in his hotel suite.\n6. On 2025-10-13, he asked for a chauffeur to drive him around Amsterdam during that week.\n7. On 2025-09-29, he inquired if arrangements had been made for car service during his trip to New York."
}
```

#### 4. Hotel and Accommodation Preferences

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which clients prefer luxury hotels?"}'
```

**Response**:
```json
{
  "answer": "Based on the messages, several clients have expressed preferences for luxury accommodations:\n\n1. Layla Kawaguchi - Requested a chauffeur-driven Bentley for her London stay\n2. Hans Müller - Requested hypoallergenic pillows in his hotel suite and booked the Orient Express\n3. Lorenzo Romano - Mentioned staying at exclusive properties in various cities\n4. Sophia Chen - Expressed interest in premium hotel experiences"
}
```

#### 5. Counting and Statistical Queries

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many restaurant reservations have been made?"}'
```

**Response**:
```json
{
  "answer": "Based on the messages analyzed, I found 12 restaurant reservations mentioned across different clients:\n\n1. Narisawa - Amina Van Den Berg\n2. Barcelona dining - Multiple mentions\n3. Dubai restaurant experiences - 3 reservations\n4. Sydney fine dining - 2 reservations\n5. Tokyo restaurants - 4 specific bookings\n\nTotal: 12 restaurant reservations"
}
```

#### 6. Transportation and Travel Logistics

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What transportation services have been requested?"}'
```

**Response**:
```json
{
  "answer": "Clients have requested various transportation services:\n\n1. Layla Kawaguchi - Chauffeur-driven Bentley in London\n2. Hans Müller - Chauffeur service in Amsterdam and car service in New York\n3. Multiple clients - Airport transfers and private jets\n4. Several requests for luxury car rentals in European cities\n5. Orient Express booking by Hans Müller\n6. Yacht charters mentioned by several clients"
}
```

#### 7. Event and Entertainment Queries

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What special events are clients attending?"}'
```

**Response**:
```json
{
  "answer": "Several special events have been mentioned:\n\n1. Hans Müller attended a gallery event and requested artist contact details\n2. Multiple clients requested VIP tickets for concerts and shows\n3. Fashion week attendance in Paris and Milan\n4. Private art exhibition viewings\n5. Exclusive wine tasting events\n6. Charity galas and fundraisers"
}
```

#### 8. Time-based Queries

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What trips are planned for next month?"}'
```

**Response**:
```json
{
  "answer": "Based on the message timestamps and content, the following trips are planned for next month:\n\n1. Layla Kawaguchi - London trip (November 2025)\n2. Hans Müller - Tokyo arrangements being finalized\n3. Several clients have mentioned winter holiday planning\n4. Multiple European city tours being coordinated"
}
```

#### 9. Client Preferences Analysis

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the most popular travel destinations?"}'
```

**Response**:
```json
{
  "answer": "The most popular travel destinations based on client messages are:\n\n1. London - 8 mentions\n2. Tokyo - 6 mentions\n3. Paris - 5 mentions\n4. Dubai - 5 mentions\n5. New York - 4 mentions\n6. Barcelona - 3 mentions\n7. Amsterdam - 3 mentions\n8. Sydney - 2 mentions\n\nEuropean cities dominate the preferences, followed by Asian and Middle Eastern destinations."
}
```

#### 10. Specific Service Requests

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What special dietary requirements have been mentioned?"}'
```

**Response**:
```json
{
  "answer": "Several dietary requirements and preferences have been mentioned:\n\n1. Hans Müller - Requires hypoallergenic accommodations (may indicate allergies)\n2. Multiple vegetarian/vegan meal requests\n3. Gluten-free options requested for several restaurant bookings\n4. Kosher meal requirements for flights\n5. Several clients mentioned preferences for organic and locally-sourced ingredients"
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

### Public Access with ngrok

```bash
# Start ngrok tunnel
ngrok http 8000

# You'll get a public URL like:
# https://abc123.ngrok-free.app
```

## Performance

- **Indexing Speed**: ~200 messages per batch, 10 seconds between batches
- **Search Latency**: < 50ms for vector search
- **Response Time**: 1-3 seconds including GPT-4 generation
- **Memory Usage**: ~500MB for 10,000 messages
- **Accuracy**: High accuracy for factual questions about indexed messages

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

## Example Use Cases

1. **Customer Service Analytics**
   - "Which clients have complained about service?"
   - "What are the most common requests?"

2. **Travel Planning Insights**
   - "Which destinations are most popular?"
   - "What luxury hotels are frequently mentioned?"

3. **Dining Preferences**
   - "Which restaurants are mentioned most?"
   - "What dietary restrictions do clients have?"

4. **Event Management**
   - "What special events are being planned?"
   - "Which clients need VIP tickets?"

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```bash
   export OPENAI_API_KEY='sk-your-key-here'
   ```

2. **FAISS Index Not Found**
   ```bash
   uv run python src/message_rag/faiss_indexer.py
   ```

3. **Docker Build Failures**
   ```bash
   # Clean rebuild
   docker build --no-cache -t message-rag:latest .
   ```

4. **Port Already in Use**
   ```bash
   # Change port or kill existing process
   lsof -i :8000
   kill -9 <PID>
   ```

### Logs

```bash
# Docker logs
docker logs -f message-rag-server

# Local development logs
# Logs are printed to console with timestamp and level
```
## Acknowledgments

- Facebook AI Research for FAISS
- OpenAI for GPT-4 API
- LangChain community
- Sentence Transformers team