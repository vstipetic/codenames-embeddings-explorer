"""
Find the best hint word for a Codenames board configuration.

This module implements the hint-finding algorithm based on embedding similarity
and cliff threshold filtering.

Optimized for large vocabularies (100k+ words) using:
- FAISS for nearest neighbor candidate filtering
- Vectorized NumPy matrix operations (BLAS-accelerated)
- Early rejection via category mask prefiltering
"""

import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Generator

import faiss
import numpy as np


@dataclass
class TimingResult:
    """Result of a timed operation."""

    name: str
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.elapsed_seconds * 1000:.2f}ms"


@contextmanager
def timer(name: str = "operation") -> Generator[TimingResult, None, None]:
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    result = TimingResult(name=name)
    try:
        yield result
    finally:
        result.elapsed_seconds = time.perf_counter() - start


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


def load_faiss_index(index_path: str) -> faiss.IndexFlatIP:
    """Load a FAISS index from disk."""
    return faiss.read_index(index_path)


def load_hint_words_order(order_path: str) -> list[str]:
    """Load the hint words order list for FAISS index mapping."""
    with open(order_path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Vectorized helper functions for optimized hint finding
# =============================================================================


def build_board_matrix(
    board_words: list[BoardWord],
    board_embeddings: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, BoardWord]]]:
    """
    Build a contiguous matrix of board word embeddings.

    Returns:
        board_matrix: Shape (n_board, dim), dtype float32, C-contiguous
        board_word_mapping: List of (matrix_row_index, BoardWord) for valid words
    """
    valid_embeddings: list[np.ndarray] = []
    board_word_mapping: list[tuple[int, BoardWord]] = []

    for bw in board_words:
        # Handle case variations
        emb = None
        for variant in [bw.word.upper(), bw.word.lower(), bw.word]:
            if variant in board_embeddings:
                emb = board_embeddings[variant]
                break

        if emb is not None:
            board_word_mapping.append((len(valid_embeddings), bw))
            valid_embeddings.append(emb)

    if not valid_embeddings:
        return np.empty((0, 1536), dtype=np.float32), []

    # Stack into contiguous C-order array for optimal BLAS performance
    board_matrix = np.ascontiguousarray(np.vstack(valid_embeddings), dtype=np.float32)

    return board_matrix, board_word_mapping


def create_category_masks(
    board_word_mapping: list[tuple[int, BoardWord]],
    n_board: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create boolean masks for team and non-team words.

    Returns:
        team_mask: Boolean array, True for TEAM words
        non_team_mask: Boolean array, True for non-TEAM words
    """
    team_mask = np.zeros(n_board, dtype=bool)
    non_team_mask = np.ones(n_board, dtype=bool)

    for idx, bw in board_word_mapping:
        if bw.category == WordCategory.TEAM:
            team_mask[idx] = True
            non_team_mask[idx] = False

    return team_mask, non_team_mask


def prefilter_hints(
    similarity_matrix: np.ndarray,
    team_mask: np.ndarray,
    non_team_mask: np.ndarray,
) -> np.ndarray:
    """
    Find indices of hints where max team similarity > max non-team similarity.

    This is a necessary condition for a valid hint (top word must be TEAM).

    Returns:
        valid_hint_indices: Array of row indices that pass the filter
    """
    # Max similarity to any team word for each hint
    max_team_sims = similarity_matrix[:, team_mask].max(axis=1)

    # Max similarity to any non-team word for each hint
    max_non_team_sims = similarity_matrix[:, non_team_mask].max(axis=1)

    # Hint is valid only if top word could be a team word
    valid_mask = max_team_sims > max_non_team_sims

    return np.where(valid_mask)[0]


def evaluate_hint_from_similarities(
    hint_word: str,
    similarities: np.ndarray,
    board_word_mapping: list[tuple[int, BoardWord]],
    cliff_threshold: float,
) -> HintCandidate | None:
    """
    Evaluate a hint given its precomputed similarities to board words.
    """
    # Get sorted indices (descending by similarity)
    sorted_indices = np.argsort(similarities)[::-1]

    # Map matrix indices back to BoardWords
    idx_to_bw = {idx: bw for idx, bw in board_word_mapping}

    # Build sorted list of (BoardWord, similarity)
    sorted_pairs: list[tuple[BoardWord, float]] = []
    for matrix_idx in sorted_indices:
        if matrix_idx in idx_to_bw:
            sorted_pairs.append((idx_to_bw[matrix_idx], float(similarities[matrix_idx])))

    if not sorted_pairs:
        return None

    # Check if top word is TEAM
    if sorted_pairs[0][0].category != WordCategory.TEAM:
        return None

    # Count consecutive team words from top and find cliff
    team_words: list[str] = []
    last_team_idx = 0

    for idx, (bw, _sim) in enumerate(sorted_pairs):
        if bw.category == WordCategory.TEAM:
            team_words.append(bw.word)
            last_team_idx = idx
        else:
            break

    if not team_words:
        return None

    # Calculate cliff
    if last_team_idx + 1 < len(sorted_pairs):
        cliff = sorted_pairs[last_team_idx][1] - sorted_pairs[last_team_idx + 1][1]
    else:
        cliff = 1.0

    if cliff < cliff_threshold:
        return None

    return HintCandidate(
        hint_word=hint_word,
        num_words=len(team_words),
        cliff_size=cliff,
        team_words=team_words,
    )


def get_faiss_candidates(
    team_word_embeddings: np.ndarray,
    hint_index: faiss.IndexFlatIP,
    top_k_per_team: int,
) -> set[int]:
    """
    Use FAISS to find candidate hint indices that are similar to team words.

    Args:
        team_word_embeddings: Shape (n_team, dim) embeddings of team words
        hint_index: FAISS index built on hint embeddings
        top_k_per_team: Number of nearest hints to retrieve per team word

    Returns:
        Set of candidate hint indices (union across all team words)
    """
    # Search for top-k similar hints for each team word
    _distances, indices = hint_index.search(team_word_embeddings, top_k_per_team)

    # Union all candidate indices (flatten and convert to set)
    # Filter out -1 which indicates no result
    candidate_indices = set(idx for idx in indices.flatten() if idx >= 0)

    return candidate_indices


def find_best_hints(
    board_words: list[BoardWord],
    hint_embeddings: dict[str, np.ndarray],
    board_embeddings: dict[str, np.ndarray],
    hint_index: faiss.IndexFlatIP,
    hint_words_order: list[str],
    cliff_threshold: float = 0.1,
    top_k: int = 5,
    top_k_per_team: int = 5000,
    verbose: bool = False,
) -> list[HintCandidate]:
    """
    Find the best hints for a given board configuration using FAISS.

    Args:
        board_words: List of words on the board with their categories
        hint_embeddings: Dict of hint word -> embedding
        board_embeddings: Dict of board word -> embedding
        hint_index: FAISS index for candidate filtering
        hint_words_order: Word list matching FAISS index order
        cliff_threshold: Minimum similarity gap required
        top_k: Number of top hints to return
        top_k_per_team: Number of candidates to retrieve per team word from FAISS
        verbose: If True, print timing information

    Returns:
        List of top HintCandidates sorted by num_words (desc), then cliff_size (desc)
    """
    timings: list[str] = []

    # Step 1: Build excluded words set
    with timer("build_excluded_set") as t:
        excluded_words = set()
        for bw in board_words:
            excluded_words.add(bw.word.lower())
            excluded_words.add(bw.word.upper())
    timings.append(str(t))

    # Step 2: Build board matrix
    with timer("build_board_matrix") as t:
        board_matrix, board_word_mapping = build_board_matrix(board_words, board_embeddings)
    timings.append(str(t))

    if board_matrix.size == 0:
        return []

    n_board = board_matrix.shape[0]

    # Step 3: Create category masks
    with timer("create_category_masks") as t:
        team_mask, non_team_mask = create_category_masks(board_word_mapping, n_board)
    timings.append(str(t))

    # Step 4: Get team word embeddings for FAISS query
    team_indices = [idx for idx, bw in board_word_mapping if bw.category == WordCategory.TEAM]
    team_embeddings = board_matrix[team_indices]

    # Step 5: FAISS candidate search
    with timer("faiss_candidate_search") as t:
        candidate_indices = get_faiss_candidates(team_embeddings, hint_index, top_k_per_team)

        # Map FAISS indices back to hint words, filtering out board words
        candidate_hint_words = []
        candidate_hint_vectors = []
        for idx in candidate_indices:
            if idx < len(hint_words_order):
                word = hint_words_order[idx]
                if word.lower() not in excluded_words and word.upper() not in excluded_words:
                    if word in hint_embeddings:
                        candidate_hint_words.append(word)
                        candidate_hint_vectors.append(hint_embeddings[word])
    timings.append(str(t))

    # Step 6: Build candidate matrix
    with timer("build_candidate_matrix") as t:
        if not candidate_hint_vectors:
            return []
        hint_matrix = np.ascontiguousarray(
            np.vstack(candidate_hint_vectors), dtype=np.float32
        )
        hint_words = candidate_hint_words
    timings.append(str(t))

    # Step 7: Compute all similarities at once (BLAS-accelerated)
    with timer("compute_similarity_matrix") as t:
        similarity_matrix = hint_matrix @ board_matrix.T
    timings.append(str(t))

    # Step 8: Prefilter hints (vectorized early rejection)
    with timer("prefilter_hints") as t:
        valid_hint_indices = prefilter_hints(similarity_matrix, team_mask, non_team_mask)
    timings.append(str(t))

    # Step 9: Evaluate valid hints
    candidates: list[HintCandidate] = []
    with timer("evaluate_valid_hints") as t:
        for hint_idx in valid_hint_indices:
            candidate = evaluate_hint_from_similarities(
                hint_word=hint_words[hint_idx],
                similarities=similarity_matrix[hint_idx],
                board_word_mapping=board_word_mapping,
                cliff_threshold=cliff_threshold,
            )
            if candidate is not None:
                candidates.append(candidate)
    timings.append(str(t))

    # Step 10: Sort and return top_k
    with timer("sort_candidates") as t:
        candidates.sort(key=lambda c: (c.num_words, c.cliff_size), reverse=True)
    timings.append(str(t))

    if verbose:
        print("\n".join(timings))
        print(f"Total hints in vocab: {len(hint_embeddings)}")
        print(f"FAISS candidates: {len(hint_words)}")
        print(f"After prefilter: {len(valid_hint_indices)}")
        print(f"Valid candidates: {len(candidates)}")

    return candidates[:top_k]
