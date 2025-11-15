import faiss
import numpy as np
import requests
import json
import pickle
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from sentence_transformers import SentenceTransformer

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
    """Fetches all member messages with rate limiting"""

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
            batch_size: Number of records per request
            delay_seconds: Delay between requests to avoid rate limiting
        """
        self.base_url = base_url
        self.messages_endpoint = f"{base_url}/messages/"
        self.batch_size = batch_size
        self.delay_seconds = delay_seconds

    def fetch_all_messages(self) -> List[Message]:
        """Fetch ALL messages with rate limiting"""
        all_messages = []
        skip = 0
        batch_num = 1
        consecutive_errors = 0
        max_consecutive_errors = 3

        headers = {"Accept": "application/json", "User-Agent": "message-rag/1.0"}

        logger.info(
            f"Starting to fetch ALL messages (batch_size={self.batch_size}, delay={self.delay_seconds}s)"
        )
        logger.info("This may take several minutes due to rate limiting...")

        while True:
            try:
                logger.info(f"\n--- Batch {batch_num} ---")
                logger.info(f"Fetching records {skip} to {skip + self.batch_size}...")

                response = requests.get(
                    self.messages_endpoint,
                    params={"skip": skip, "limit": self.batch_size},
                    headers=headers,
                    timeout=30,
                )

                # Handle rate limiting
                if response.status_code == 402:
                    wait_time = self.delay_seconds * 2
                    logger.warning(f"Rate limit hit (402). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue

                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("Too many consecutive errors. Stopping.")
                        break
                    time.sleep(self.delay_seconds)
                    continue

                consecutive_errors = 0
                data = response.json()

                # Check if we have more data
                if "items" not in data or len(data["items"]) == 0:
                    logger.info("No more messages to fetch.")
                    break

                # Process messages
                batch_messages = [Message.from_api_response(item) for item in data["items"]]
                all_messages.extend(batch_messages)

                batch_size_actual = len(batch_messages)
                total_fetched = len(all_messages)

                logger.info(f"✓ Fetched {batch_size_actual} messages (Total: {total_fetched})")

                # Show progress if total is available
                if "total" in data:
                    total_available = data["total"]
                    progress = 100 * total_fetched / total_available
                    logger.info(f"Progress: {total_fetched}/{total_available} ({progress:.1f}%)")

                    if total_fetched >= total_available:
                        logger.info("All messages fetched!")
                        break

                # Check if this was the last batch
                if batch_size_actual < self.batch_size:
                    logger.info("Reached end of data.")
                    break

                # Prepare for next batch
                skip += self.batch_size
                batch_num += 1

                # Rate limiting delay
                logger.info(f"Waiting {self.delay_seconds} seconds before next batch...")
                time.sleep(self.delay_seconds)

            except requests.exceptions.Timeout:
                logger.error("Request timeout. Retrying...")
                time.sleep(self.delay_seconds)
                continue

            except Exception as e:
                logger.error(f"Error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many errors. Stopping.")
                    break
                time.sleep(self.delay_seconds)
                continue

        logger.info(f"Successfully fetched {len(all_messages)} total messages")

        return all_messages


class FAISSIndexBuilder:
    """Builds and manages FAISS index for message embeddings"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        data_dir: str = "data",
    ):
        """Initialize the index builder"""
        self.model_name = model_name
        self.dimension = dimension
        self.data_dir = Path(data_dir)
        self.index_dir = self.data_dir / "faiss_index"

        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        logger.info(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)

        self.index = None
        self.messages = []
        self.id_to_index = {}

    def create_embeddings(self, messages: List[Message]) -> np.ndarray:
        """Create embeddings for all messages"""
        logger.info(f"Creating embeddings for {len(messages)} messages...")

        texts = []
        for msg in messages:
            # Rich context for better retrieval
            text = f"{msg.message} [User: {msg.user_name}] [Date: {msg.timestamp[:10]}]"
            texts.append(text)

        # Create embeddings with progress bar
        embeddings = self.encoder.encode(
            texts, show_progress_bar=True, convert_to_numpy=True, batch_size=32
        )

        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings

    def build_index(self, messages: List[Message]) -> None:
        """Build FAISS index from messages"""
        if not messages:
            logger.warning("No messages to index")
            return

        embeddings = self.create_embeddings(messages)

        self.messages = messages
        self.id_to_index = {msg.id: idx for idx, msg in enumerate(messages)}

        logger.info("Building FAISS index...")

        # Use IVF index for large datasets (>10k messages)
        if len(messages) > 10000:
            logger.info("Using IVF index for large dataset...")
            nlist = min(100, len(messages) // 100)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)
            self.index.add(embeddings)
        else:
            # Use flat index for smaller datasets
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)

        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")

    def search(
        self, query: str, k: int = 10, user_filter: Optional[str] = None
    ) -> List[Tuple[Message, float]]:
        """Search for similar messages"""
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []

        # Encode and normalize query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search (get more results for filtering)
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

        # Save human-readable summary
        self._save_summary(stats)

        logger.info("Index saved successfully")

    def load_index(self) -> bool:
        """Load index from disk"""
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

            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return True

        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the index"""
        if not self.messages:
            return {"status": "empty"}

        # Group messages by user
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

    def _save_summary(self, stats: Dict[str, Any]) -> None:
        """Save human-readable summary"""
        with open(self.index_dir / "summary.txt", "w") as f:
            f.write("FAISS Index Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Created: {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Dimension: {self.dimension}\n")
            f.write(f"Total messages: {stats['total_messages']}\n")
            f.write(f"Total users: {stats['total_users']}\n")
            f.write(f"Index type: {stats['index_type']}\n")
            f.write(f"\nUser distribution:\n")

            for user, count in sorted(stats["users"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {user}: {count} messages\n")


def main():
    """Main function to build or load index"""
    logger.info("FAISS Index Builder")

    builder = FAISSIndexBuilder()

    # Check if index already exists
    if builder.index_dir.exists() and (builder.index_dir / "index.bin").exists():
        logger.info("Found existing index.")
        if builder.load_index():
            stats = builder.get_statistics()
            logger.info(f"Using existing index with {stats['total_messages']} messages")

            response = input("\nRebuild index with fresh data? (y/n): ")
            if response.lower() != "y":
                return builder

    # Fetch all messages
    fetcher = MemberDataFetcher(batch_size=200, delay_seconds=10)
    messages = fetcher.fetch_all_messages()

    if not messages:
        logger.error("No messages fetched from API")
        return None

    # Build and save index
    builder.build_index(messages)
    builder.save_index()

    # Print statistics
    stats = builder.get_statistics()
    logger.info("\nIndex Statistics:")
    logger.info(json.dumps(stats, indent=2))

    return builder


if __name__ == "__main__":
    builder = main()
