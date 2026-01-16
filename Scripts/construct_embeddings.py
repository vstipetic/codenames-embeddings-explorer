"""
Construct embeddings for hint vocabulary and Codenames words.

Usage:
    uv run python Scripts/construct_embeddings.py
"""

import os
import pickle
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


def load_words_from_file(filepath: str) -> list[str]:
    """Load words from a text file, one word per line."""
    with open(filepath, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def get_embeddings_batch(
    client: OpenAI,
    words: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> dict[str, np.ndarray]:
    """
    Get embeddings for a list of words from OpenAI API.

    Batches requests to avoid rate limits.
    Returns dict mapping word -> normalized embedding vector.
    """
    embeddings = {}

    for i in range(0, len(words), batch_size):
        batch = words[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(words) - 1) // batch_size + 1}...")

        response = client.embeddings.create(input=batch, model=model)

        for word, data in zip(batch, response.data):
            vec = np.array(data.embedding, dtype=np.float32)
            # Normalize for cosine similarity via dot product
            vec = vec / np.linalg.norm(vec)
            embeddings[word] = vec

    return embeddings


def save_embeddings(embeddings: dict[str, np.ndarray], filepath: str) -> None:
    """Save embeddings dictionary to pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved {len(embeddings)} embeddings to {filepath}")


def main():
    """Generate embeddings for hint vocabulary and Codenames words."""
    # Load from .env if it exists, otherwise use system environment variable
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set it as a system variable or in a .env file.")

    client = OpenAI(api_key=api_key)

    # Paths
    hint_words_path = "Data/common_english_words.txt"
    codenames_words_path = "Data/codenames_words.txt"
    hint_embeddings_path = "Storage/hint_embeddings.pkl"
    codenames_embeddings_path = "Storage/codenames_embeddings.pkl"

    # Load words
    print("Loading word lists...")
    hint_words = load_words_from_file(hint_words_path)
    codenames_words = load_words_from_file(codenames_words_path)

    print(f"Loaded {len(hint_words)} hint words")
    print(f"Loaded {len(codenames_words)} Codenames words")

    # Generate embeddings for hint vocabulary
    print("\nGenerating embeddings for hint vocabulary...")
    hint_embeddings = get_embeddings_batch(client, hint_words)
    save_embeddings(hint_embeddings, hint_embeddings_path)

    # Generate embeddings for Codenames words
    print("\nGenerating embeddings for Codenames words...")
    codenames_embeddings = get_embeddings_batch(client, codenames_words)
    save_embeddings(codenames_embeddings, codenames_embeddings_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
