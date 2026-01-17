"""
Construct embeddings for hint vocabulary and Codenames words.

Usage:
    uv run python Scripts/construct_embeddings.py

Generates:
    - Pickle files with embeddings (dict[str, np.ndarray])
    - FAISS index for hint embeddings (for efficient nearest neighbor search)
    - Word order file for FAISS ID mapping
"""

import csv
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Retry configuration
MAX_RETRIES_PER_BATCH = 5


def load_words_from_file(filepath: str) -> list[str]:
    """Load words from a text file, one word per line, normalized to lowercase."""
    with open(filepath, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def load_words_from_csv(filepath: str, column: str = "word") -> list[str]:
    """Load words from a CSV file, extracting specified column, normalized to lowercase."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row[column].strip().lower() for row in reader if row[column].strip()]


def get_embeddings_batch(
    client: OpenAI,
    words: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 1000,
    max_retries: int = MAX_RETRIES_PER_BATCH,
) -> dict[str, np.ndarray]:
    """
    Get embeddings for a list of words from OpenAI API.

    Batches requests to avoid rate limits.
    Retries failed batches with exponential backoff.
    Returns dict mapping word -> normalized embedding vector.
    """
    embeddings = {}

    batch_ranges = range(0, len(words), batch_size)
    total_batches = (len(words) - 1) // batch_size + 1

    for i in tqdm(batch_ranges, total=total_batches, desc="Embedding batches"):
        batch = words[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                response = client.embeddings.create(input=batch, model=model)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # 1, 2, 4, 8, 16 seconds
                    print(f"\nBatch {batch_num}/{total_batches} failed (attempt {attempt + 1}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"Batch {batch_num}/{total_batches} failed after {max_retries} attempts: {e}"
                    )

        for word, data in zip(batch, response.data):
            vec = np.array(data.embedding, dtype=np.float32)
            # Zero-norm guard: prevent NaN from division by zero
            norm = np.linalg.norm(vec)
            if norm == 0:
                raise ValueError(f"Zero-norm embedding returned for word: '{word}'")
            vec = vec / norm
            embeddings[word] = vec

    return embeddings


def save_embeddings(embeddings: dict[str, np.ndarray], filepath: str) -> None:
    """Save embeddings dictionary to pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved {len(embeddings)} embeddings to {filepath}")


def save_missing_words(missing_words: list[str], filepath: str) -> None:
    """Save missing words (failed embeddings) to a text file, one per line."""
    if not missing_words:
        return
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(missing_words))
    print(f"Saved {len(missing_words)} missing words to {filepath}")


def save_faiss_index(
    embeddings: dict[str, np.ndarray],
    index_path: str,
    order_path: str,
) -> None:
    """
    Build and save a FAISS index for efficient nearest neighbor search.

    Args:
        embeddings: Dict of word -> normalized embedding
        index_path: Path to save FAISS index
        order_path: Path to save word order (for mapping FAISS IDs back to words)
    """
    # Get words and embeddings in consistent order
    words = list(embeddings.keys())
    matrix = np.vstack([embeddings[w] for w in words]).astype(np.float32)

    # Ensure contiguous for FAISS
    matrix = np.ascontiguousarray(matrix)

    # Create index using inner product (cosine similarity for normalized vectors)
    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # Save index
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index ({len(words)} vectors, dim={dim}) to {index_path}")

    # Save word order for ID mapping
    with open(order_path, "wb") as f:
        pickle.dump(words, f)
    print(f"Saved word order to {order_path}")


def main():
    """Generate embeddings for hint vocabulary and Codenames words."""
    # Load from .env if it exists, otherwise use system environment variable
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set it as a system variable or in a .env file.")

    client = OpenAI(api_key=api_key)

    # Paths
    hint_words_path = "Storage/unigram_freq.csv"
    codenames_words_path = "Storage/codename_words.txt"
    hint_embeddings_path = "Storage/hint_embeddings.pkl"
    codenames_embeddings_path = "Storage/codenames_embeddings.pkl"
    hint_faiss_index_path = "Storage/hint_embeddings_faiss.index"
    hint_words_order_path = "Storage/hint_words_order.pkl"
    hint_missing_words_path = "Storage/hint_missing_words.txt"
    codenames_missing_words_path = "Storage/codenames_missing_words.txt"

    # Load words
    print("Loading word lists...")
    hint_words = load_words_from_csv(hint_words_path, column="word")
    codenames_words = load_words_from_file(codenames_words_path)

    print(f"Loaded {len(hint_words)} hint words")
    print(f"Loaded {len(codenames_words)} Codenames words")

    # Generate embeddings for hint vocabulary
    print("\nGenerating embeddings for hint vocabulary...")
    hint_embeddings = get_embeddings_batch(client, hint_words)
    save_embeddings(hint_embeddings, hint_embeddings_path)
    print(f"Hint embeddings stored: {len(hint_embeddings)}")
    hint_missing_words = [w for w in hint_words if w not in hint_embeddings]
    save_missing_words(hint_missing_words, hint_missing_words_path)

    # Generate FAISS index for hint vocabulary (for efficient nearest neighbor search)
    print("\nGenerating FAISS index for hint vocabulary...")
    save_faiss_index(hint_embeddings, hint_faiss_index_path, hint_words_order_path)

    # Generate embeddings for Codenames words
    print("\nGenerating embeddings for Codenames words...")
    codenames_embeddings = get_embeddings_batch(client, codenames_words)
    save_embeddings(codenames_embeddings, codenames_embeddings_path)
    print(f"Codenames embeddings stored: {len(codenames_embeddings)}")
    codenames_missing_words = [w for w in codenames_words if w not in codenames_embeddings]
    save_missing_words(codenames_missing_words, codenames_missing_words_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
