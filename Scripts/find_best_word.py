"""
Find the best hint word for a Codenames board configuration.

This module implements the hint-finding algorithm based on embedding similarity
and cliff threshold filtering.
"""

import pickle
from dataclasses import dataclass
from enum import Enum

import numpy as np


class WordCategory(Enum):
    """Categories for words on a Codenames board."""

    TEAM = "team"
    ENEMY = "enemy"
    TRAP = "trap"
    NEUTRAL = "neutral"


@dataclass
class BoardWord:
    """A word on the Codenames board with its category."""

    word: str
    category: WordCategory


@dataclass
class HintCandidate:
    """A potential hint with its evaluation metrics."""

    hint_word: str
    num_words: int  # Number of consecutive team words from top
    cliff_size: float  # Similarity gap to first non-team word
    team_words: list[str]  # The team words this hint covers


def load_embeddings(filepath: str) -> dict[str, np.ndarray]:
    """Load embeddings from pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Assumes vectors are already normalized (dot product = cosine similarity).
    """
    return float(np.dot(vec1, vec2))


def evaluate_hint(
    hint_word: str,
    hint_embedding: np.ndarray,
    board_words: list[BoardWord],
    board_embeddings: dict[str, np.ndarray],
    cliff_threshold: float,
) -> HintCandidate | None:
    """
    Evaluate a potential hint word against the board.

    Algorithm:
    1. Calculate similarity to all board words
    2. Sort by similarity descending
    3. If top word is not a team word, discard
    4. Count consecutive team words from top
    5. Calculate cliff (gap to first non-team word)
    6. If cliff < threshold, discard

    Returns HintCandidate if valid, None otherwise.
    """
    # Calculate similarities to all board words
    similarities: list[tuple[BoardWord, float]] = []
    for bw in board_words:
        if bw.word.upper() in board_embeddings:
            emb = board_embeddings[bw.word.upper()]
        elif bw.word.lower() in board_embeddings:
            emb = board_embeddings[bw.word.lower()]
        elif bw.word in board_embeddings:
            emb = board_embeddings[bw.word]
        else:
            continue  # Skip if no embedding found
        sim = cosine_similarity(hint_embedding, emb)
        similarities.append((bw, sim))

    if not similarities:
        return None

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Check if top word is a team word
    if similarities[0][0].category != WordCategory.TEAM:
        return None

    # Count consecutive team words from top
    team_words = []
    last_team_idx = 0
    for idx, (bw, _sim) in enumerate(similarities):
        if bw.category == WordCategory.TEAM:
            team_words.append(bw.word)
            last_team_idx = idx
        else:
            break

    num_words = len(team_words)
    if num_words == 0:
        return None

    # Calculate cliff (gap to first non-team word)
    if last_team_idx + 1 < len(similarities):
        cliff = similarities[last_team_idx][1] - similarities[last_team_idx + 1][1]
    else:
        # All words are team words - maximum cliff
        cliff = 1.0

    # Check cliff threshold
    if cliff < cliff_threshold:
        return None

    return HintCandidate(
        hint_word=hint_word,
        num_words=num_words,
        cliff_size=cliff,
        team_words=team_words,
    )


def find_best_hints(
    board_words: list[BoardWord],
    hint_embeddings: dict[str, np.ndarray],
    board_embeddings: dict[str, np.ndarray],
    cliff_threshold: float = 0.1,
    top_k: int = 5,
) -> list[HintCandidate]:
    """
    Find the best hints for a given board configuration.

    Args:
        board_words: List of words on the board with their categories
        hint_embeddings: Dict of hint word -> embedding
        board_embeddings: Dict of board word -> embedding
        cliff_threshold: Minimum similarity gap required
        top_k: Number of top hints to return

    Returns:
        List of top HintCandidates sorted by num_words (desc), then cliff_size (desc)
    """
    # Get board word strings (both cases) to filter out from hints
    board_word_set = set()
    for bw in board_words:
        board_word_set.add(bw.word.lower())
        board_word_set.add(bw.word.upper())

    candidates: list[HintCandidate] = []

    for hint_word, hint_emb in hint_embeddings.items():
        # Skip if hint word appears on board
        if hint_word.lower() in board_word_set or hint_word.upper() in board_word_set:
            continue

        candidate = evaluate_hint(
            hint_word=hint_word,
            hint_embedding=hint_emb,
            board_words=board_words,
            board_embeddings=board_embeddings,
            cliff_threshold=cliff_threshold,
        )

        if candidate is not None:
            candidates.append(candidate)

    # Sort by num_words (desc), then cliff_size (desc)
    candidates.sort(key=lambda c: (c.num_words, c.cliff_size), reverse=True)

    return candidates[:top_k]
