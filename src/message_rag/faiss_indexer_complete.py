import faiss
import numpy as np
import requests
import json
import pickle
from datetime import datetime
import time
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Data class for member messages"""

    id: str
    user_id: str
    user_name: str
    timestamp: str
    message: str
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding embedding"""
        data = asdict(self)
        data.pop("embedding", None)
        return data

    @classmethod
    def from_api_response(cls, data: Dict) -> "Message":
        """Create Message from API response"""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            user_name=data["user_name"],
            timestamp=data["timestamp"],
            message=data["message"],
        )


class MemberDataFetcher:
    """Fetches ALL member messages with proper rate limiting"""

    def __init__(
        self,
        base_url: str = "https://november7-730026606190.europe-west1.run.app",
        batch_size: int = 200,
        delay_seconds: int = 10,
    ):
        """
        Initialize fetcher with rate limiting

        Args:
            base_url: API base URL
            batch_size: Number of records per request (200)
            delay_seconds: Delay between requests (10 seconds)
        """
        self.base_url = base_url
        self.messages_endpoint = f"{base_url}/messages/"
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds

    def fetch_all_messages(self) -> List[Message]:
        """
        Fetch ALL messages with rate limiting
        """
        all_messages = []
        skip = 0
        batch_num = 1
        total_fetched = 0
        consecutive_errors = 0
        max_consecutive_errors = 3

        # Headers to avoid 401/403 errors
        headers = {"Accept": "application/json", "User-Agent": "message-rag/1.0"}

        logger.info(
            f"Starting to fetch ALL messages (batch size: {self.batch_size}, delay: {self.delay_seconds}s)"
        )
        logger.info("This may take several minutes...")

        while True:
            try:
                logger.info(f"\n--- Batch {batch_num} ---")
                logger.info(f"Fetching records {skip} to {skip + self.batch_size}...")

                # Make request
                response = requests.get(
                    self.messages_endpoint,
                    params={"skip": skip, "limit": self.batch_size},
                    headers=headers,
                    timeout=30,
                )

                # Check for rate limiting
                if response.status_code == 402:
                    logger.warning(
                        f"Rate limit hit (402). Waiting {self.delay_seconds * 2} seconds..."
                    )
                    time.sleep(self.delay_seconds * 2)
                    continue

                # Check for other errors
                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Too many consecutive errors. Stopping.")
                        break
                    time.sleep(self.delay_seconds)
                    continue

                # Reset error counter on success
                consecutive_errors = 0

                # Parse response
                data = response.json()

                # Check if we have items
                if "items" not in data or len(data["items"]) == 0:
                    logger.info("No more messages to fetch.")
                    break

                # Process messages
                batch_messages = [Message.from_api_response(item) for item in data["items"]]
                all_messages.extend(batch_messages)

                batch_size_actual = len(batch_messages)
                total_fetched += batch_size_actual

                logger.info(f"✓ Fetched {batch_size_actual} messages (Total: {total_fetched})")

                # Check if we've fetched all available
                if "total" in data:
                    total_available = data["total"]
                    logger.info(
                        f"Progress: {total_fetched}/{total_available} ({100*total_fetched/total_available:.1f}%)"
                    )

                    if total_fetched >= total_available:
                        logger.info("✅ All messages fetched!")
                        break

                # If batch is less than requested, we've reached the end
                if batch_size_actual < self.batch_size:
                    logger.info("Reached end of data (partial batch)")
                    break

                # Update for next iteration
                skip += self.batch_size
                batch_num += 1

                # Rate limiting delay
                logger.info(f"Waiting {self.delay_seconds} seconds before next batch...")
                time.sleep(self.delay_seconds)

            except requests.exceptions.Timeout:
                logger.error("Request timeout. Retrying...")
                time.sleep(self.delay_seconds)
                continue

            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many errors. Stopping.")
                    break
                time.sleep(self.delay_seconds)
                continue

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break

        logger.info("=" * 60)
        logger.info(f"Successfully fetched {len(all_messages)} total messages")
        logger.info("=" * 60)

        return all_messages


class FAISSIndexBuilder:
    """Builds and manages FAISS index for message embeddings"""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384, data_dir: str = "data"
    ):
        """Initialize the index builder"""
        self.model_name = model_name
        self.dimension = dimension
        self.data_dir = Path(data_dir)
        self.index_dir = (
            self.data_dir / "faiss_index_complete"
        )  # Different directory for complete index

        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        # Initialize FAISS index
        self.index = None
        self.messages = []
        self.id_to_index = {}

    def create_embeddings(self, messages: List[Message]) -> np.ndarray:
        """Create embeddings for all messages"""
        logger.info(f"Creating embeddings for {len(messages)} messages...")
        logger.info("This may take a few minutes for large datasets...")

        # Prepare texts
        texts = []
        for msg in messages:
            # Rich context for better retrieval
            text = f"{msg.message} [User: {msg.user_name}] [Date: {msg.timestamp[:10]}]"
            texts.append(text)

        # Batch encode with progress bar
        batch_size = 32
        embeddings = self.encoder.encode(
            texts, show_progress_bar=True, convert_to_numpy=True, batch_size=batch_size
        )

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        logger.info(f"✓ Created embeddings with shape: {embeddings.shape}")
        return embeddings

    def build_index(self, messages: List[Message]) -> None:
        """Build FAISS index from messages"""
        if not messages:
            logger.warning("No messages to index")
            return

        # Create embeddings
        embeddings = self.create_embeddings(messages)

        # Store messages and create ID mapping
        self.messages = messages
        self.id_to_index = {msg.id: idx for idx, msg in enumerate(messages)}

        # Create FAISS index
        logger.info("Building FAISS index...")

        # For larger datasets, consider using IVF index for faster search
        if len(messages) > 10000:
            # Use IVF index for large datasets
            logger.info("Using IVF index for large dataset...")
            nlist = min(100, len(messages) // 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)
            self.index.add(embeddings)
        else:
            # Use flat index for smaller datasets
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)

        logger.info(f"✓ Successfully built FAISS index with {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 10, user_filter: Optional[str] = None) -> List[tuple]:
        """Search for similar messages"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, min(k * 3, self.index.ntotal))

        # Collect results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.messages):
                message = self.messages[idx]

                # Apply user filter if specified
                if user_filter and message.user_name != user_filter:
                    continue

                results.append((message, float(score)))

                if len(results) >= k:
                    break

        return results

    def save_index(self) -> None:
        """Save index and metadata to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return

        logger.info(f"Saving index to {self.index_dir}...")

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_dir / "index.bin"))

        # Save metadata
        metadata = {
            "messages": [msg.to_dict() for msg in self.messages],
            "id_to_index": self.id_to_index,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "created_at": datetime.now().isoformat(),
            "message_count": len(self.messages),
        }

        with open(self.index_dir / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        # Save statistics
        stats = self.get_statistics()
        with open(self.index_dir / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Save summary
        with open(self.index_dir / "summary.txt", "w") as f:
            f.write(f"FAISS Index Summary (Complete Dataset)\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Created: {metadata['created_at']}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dimension: {self.dimension}\n")
            f.write(f"Total vectors: {self.index.ntotal}\n")
            f.write(f"Unique users: {len(set(msg.user_name for msg in self.messages))}\n")
            f.write(f"\nUser distribution:\n")
            user_counts = {}
            for msg in self.messages:
                user_counts[msg.user_name] = user_counts.get(msg.user_name, 0) + 1
            for user, count in sorted(user_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {user}: {count} messages\n")

        logger.info("Index saved successfully")

    def load_index(self, index_dir: str = None) -> bool:
        """Load index from disk"""
        if index_dir:
            self.index_dir = Path(index_dir)

        if not self.index_dir.exists():
            logger.error(f"Index directory {self.index_dir} does not exist")
            return False

        try:
            logger.info(f"Loading index from {self.index_dir}...")

            # Load FAISS index
            self.index = faiss.read_index(str(self.index_dir / "index.bin"))

            # Load metadata
            with open(self.index_dir / "metadata.pkl", "rb") as f:
                metadata = pickle.load(f)

            # Restore messages
            self.messages = [Message(**msg_dict) for msg_dict in metadata["messages"]]
            self.id_to_index = metadata["id_to_index"]
            self.model_name = metadata["model_name"]
            self.dimension = metadata["dimension"]

            logger.info(f"✓ Successfully loaded index with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        if not self.messages:
            return {"status": "empty"}

        user_messages = {}
        for msg in self.messages:
            if msg.user_name not in user_messages:
                user_messages[msg.user_name] = []
            user_messages[msg.user_name].append(msg)

        return {
            "total_messages": len(self.messages),
            "total_users": len(user_messages),
            "index_size": self.index.ntotal if self.index else 0,
            "model": self.model_name,
            "dimension": self.dimension,
            "users": {user: len(msgs) for user, msgs in user_messages.items()},
            "earliest_message": (
                min(msg.timestamp for msg in self.messages) if self.messages else None
            ),
            "latest_message": (
                max(msg.timestamp for msg in self.messages) if self.messages else None
            ),
            "index_type": "IVF" if len(self.messages) > 10000 else "Flat",
        }


def main():
    """Main function to build complete index"""
    logger.info("=" * 60)
    logger.info("COMPLETE FAISS Index Builder")
    logger.info("=" * 60)

    # Initialize components
    fetcher = MemberDataFetcher(batch_size=200, delay_seconds=10)
    builder = FAISSIndexBuilder(data_dir="data")

    # Check if complete index already exists
    if Path("data/faiss_index_complete").exists():
        logger.info("Found existing complete index.")
        if builder.load_index("data/faiss_index_complete"):
            stats = builder.get_statistics()
            logger.info(f"Loaded index with {stats['total_messages']} messages")

            response = input("\nRebuild index with fresh data? (y/n): ")
            if response.lower() != "y":
                logger.info("Using existing index.")
                return builder

    # Fetch ALL messages
    logger.info("\nFetching ALL messages from API...")
    logger.info("This will take several minutes due to rate limiting...")

    messages = fetcher.fetch_all_messages()

    if not messages:
        logger.error("No messages fetched from API")
        return None

    logger.info(f"\n✅ Fetched {len(messages)} messages total")

    # Build index
    builder.build_index(messages)

    # Save index
    builder.save_index()

    # Print statistics
    stats = builder.get_statistics()
    logger.info("\n" + "=" * 60)
    logger.info("Index Statistics:")
    logger.info(json.dumps(stats, indent=2))

    # Test searches
    logger.info("\n" + "=" * 60)
    logger.info("Testing searches on complete index:")

    test_queries = [
        "trip to London",
        "restaurant reservation",
        "luxury hotel",
        "flight booking",
        "spa services",
    ]

    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = builder.search(query, k=3)
        for msg, score in results:
            logger.info(f"  [{score:.3f}] {msg.user_name}: {msg.message[:100]}...")

    logger.info("\n" + "=" * 60)
    logger.info("Complete indexing finished!")
    logger.info(f"Index saved to: data/faiss_index_complete/")
    logger.info("=" * 60)

    return builder


if __name__ == "__main__":
    builder = main()
