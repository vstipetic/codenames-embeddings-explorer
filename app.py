"""
Streamlit app for Codenames hint generation.

Usage:
    uv run streamlit run app.py
"""

import random
from pathlib import Path

import numpy as np
import streamlit as st

from Scripts.find_best_word import (
    BoardWord,
    HintCandidate,
    WordCategory,
    find_best_hints,
    load_embeddings,
    load_faiss_index,
    load_hint_words_order,
)


@st.cache_resource
def load_all_embeddings():
    """Load embeddings and FAISS index (cached across reruns)."""
    hint_path = Path("Storage/hint_embeddings.pkl")
    codenames_path = Path("Storage/codenames_embeddings.pkl")
    faiss_index_path = Path("Storage/hint_embeddings_faiss.index")
    words_order_path = Path("Storage/hint_words_order.pkl")

    # Check all required files exist
    required_files = [hint_path, codenames_path, faiss_index_path, words_order_path]
    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        return None, None, None, None, missing

    hint_embeddings = load_embeddings(str(hint_path))
    codenames_embeddings = load_embeddings(str(codenames_path))
    hint_index = load_faiss_index(str(faiss_index_path))
    hint_words_order = load_hint_words_order(str(words_order_path))

    return hint_embeddings, codenames_embeddings, hint_index, hint_words_order, []


def load_codenames_words() -> list[str]:
    """Load Codenames word list from Storage folder."""
    path = Path("Storage/codename_words.txt")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def compute_hint_similarities(
    hint_word: str,
    board: list[BoardWord],
    hint_embeddings: dict[str, np.ndarray],
    board_embeddings: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compute cosine similarity between a hint word and all board words.

    Args:
        hint_word: The selected hint word
        board: List of board words
        hint_embeddings: Dict of hint word -> embedding
        board_embeddings: Dict of board word -> embedding

    Returns:
        Dict mapping board word -> similarity score
    """
    similarities: dict[str, float] = {}

    # Get hint embedding
    hint_emb = hint_embeddings.get(hint_word.lower())
    if hint_emb is None:
        return similarities

    for bw in board:
        # Try different case variations for board word
        board_emb = None
        for variant in [bw.word.upper(), bw.word.lower(), bw.word]:
            if variant in board_embeddings:
                board_emb = board_embeddings[variant]
                break

        if board_emb is not None:
            # Cosine similarity (embeddings are already normalized)
            sim = float(np.dot(hint_emb, board_emb))
            similarities[bw.word] = sim

    return similarities


def generate_board(words: list[str], user_team: str) -> list[BoardWord]:
    """
    Generate a random 25-word board (5x5 grid).

    Distribution:
    - 9 team words (user's team)
    - 8 enemy words
    - 7 neutral words
    - 1 trap (assassin) word
    """
    if len(words) < 25:
        st.error(f"Not enough words in list ({len(words)}). Need at least 25.")
        return []

    selected = random.sample(words, 25)
    random.shuffle(selected)

    board: list[BoardWord] = []

    # Assign categories: 9 team, 8 enemy, 7 neutral, 1 trap
    for i, word in enumerate(selected):
        if i < 9:
            category = WordCategory.TEAM
        elif i < 17:
            category = WordCategory.ENEMY
        elif i < 24:
            category = WordCategory.NEUTRAL
        else:
            category = WordCategory.TRAP

        board.append(BoardWord(word=word, category=category))

    # Shuffle again so categories aren't in order
    random.shuffle(board)
    return board


def get_card_style(category: WordCategory, user_team: str) -> str:
    """Get CSS style for a word card based on its category."""
    if category == WordCategory.TEAM:
        bg_color = "#dc2626" if user_team == "Red" else "#2563eb"  # Red or Blue
        text_color = "white"
    elif category == WordCategory.ENEMY:
        bg_color = "#2563eb" if user_team == "Red" else "#dc2626"  # Opposite color
        text_color = "white"
    elif category == WordCategory.NEUTRAL:
        bg_color = "#f5f5dc"  # Beige
        text_color = "black"
    else:  # TRAP
        bg_color = "#1a1a1a"  # Black
        text_color = "white"

    return f"""
        background-color: {bg_color};
        color: {text_color};
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 14px;
        min-height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
    """


def display_board(
    board: list[BoardWord],
    user_team: str,
    similarities: dict[str, float] | None = None,
):
    """
    Display the board as a 5x5 grid.

    Args:
        board: List of board words
        user_team: Current user's team color
        similarities: Optional dict mapping word -> similarity score (for selected hint)
    """
    cols_per_row = 5

    for row in range(5):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            idx = row * cols_per_row + col_idx
            if idx < len(board):
                bw = board[idx]
                style = get_card_style(bw.category, user_team)

                # Build display text with optional similarity
                if similarities and bw.word in similarities:
                    sim = similarities[bw.word]
                    display_text = f"{bw.word}<br><small>({sim:.3f})</small>"
                else:
                    display_text = bw.word

                col.markdown(
                    f'<div style="{style}">{display_text}</div>',
                    unsafe_allow_html=True,
                )


def display_hints(hints: list[HintCandidate]):
    """Display the generated hints."""
    if not hints:
        st.info("No hints found with current threshold. Try lowering the cliff threshold.")
        return

    for i, hint in enumerate(hints, 1):
        with st.expander(f"**{hint.hint_word}** - {hint.num_words} word(s)", expanded=(i == 1)):
            st.write(f"**Cliff size:** {hint.cliff_size:.3f}")
            st.write(f"**Covers:** {', '.join(hint.team_words)}")


def main():
    st.set_page_config(page_title="Codenames Hint Generator", layout="wide")
    st.title("Codenames Hint Generator")

    # Load resources
    result = load_all_embeddings()
    hint_embeddings, codenames_embeddings, hint_index, hint_words_order, missing_files = result
    codenames_words = load_codenames_words()

    # Check if all required files exist
    if missing_files:
        st.error(
            "Required files not found! Please run the embedding script first:\n\n"
            "```bash\nuv run python Scripts/construct_embeddings.py\n```\n\n"
            f"Missing files:\n" + "\n".join(f"- {f}" for f in missing_files)
        )
        return

    if not codenames_words:
        st.error("Codenames word list not found at Storage/codename_words.txt")
        return

    # Sidebar controls
    with st.sidebar:
        st.header("Controls")

        if st.button("Generate New Board", type="primary"):
            user_team = st.session_state.get("user_team", "Red")
            st.session_state.board = generate_board(codenames_words, user_team)

        user_team = st.radio("Your Team", ["Red", "Blue"], key="user_team")

        cliff_threshold = st.slider(
            "Cliff Threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.05,
            step=0.01,
            help="Minimum similarity gap between last team word and first non-team word",
        )

        # Vocabulary size selector (words are sorted by frequency in the CSV)
        total_vocab = len(hint_embeddings)
        vocab_options = [10_000, 25_000, 50_000, 100_000, total_vocab]
        vocab_options = [v for v in vocab_options if v <= total_vocab]  # Filter valid options
        if total_vocab not in vocab_options:
            vocab_options.append(total_vocab)

        vocab_size = st.select_slider(
            "Vocabulary Size",
            options=vocab_options,
            value=vocab_options[-1],  # Default to all words
            format_func=lambda x: f"{x:,}" if x < total_vocab else f"All ({x:,})",
            help="Limit hints to top N most frequent words from vocabulary",
        )

        st.markdown("---")
        st.markdown("### Legend")
        st.markdown(f"- **{'Red' if user_team == 'Red' else 'Blue'}**: Your team")
        st.markdown(f"- **{'Blue' if user_team == 'Red' else 'Red'}**: Enemy team")
        st.markdown("- **Beige**: Neutral")
        st.markdown("- **Black**: Trap (Assassin)")

        st.markdown("---")
        st.caption(f"Using top {vocab_size:,} of {total_vocab:,} words")

    # Main content
    if "board" not in st.session_state:
        st.info("Click 'Generate New Board' to start!")
        return

    board = st.session_state.board

    # Generate hints first (needed for selection)
    with st.spinner("Finding best hints..."):
        hints = find_best_hints(
            board_words=board,
            hint_embeddings=hint_embeddings,
            board_embeddings=codenames_embeddings,
            hint_index=hint_index,
            hint_words_order=hint_words_order,
            cliff_threshold=cliff_threshold,
            top_k=10,
            max_vocab_size=vocab_size,
        )

    # Hint selection
    selected_similarities: dict[str, float] | None = None
    if hints:
        st.subheader(f"Best Hints for {user_team} Team")

        # Create hint options for selectbox
        hint_options = ["(None - hide similarities)"] + [
            f"{h.hint_word} - {h.num_words} word(s)" for h in hints
        ]

        selected_idx = st.selectbox(
            "Select a hint to see similarities on board:",
            range(len(hint_options)),
            format_func=lambda i: hint_options[i],
            key="selected_hint",
        )

        # Compute similarities for selected hint
        if selected_idx > 0:
            selected_hint = hints[selected_idx - 1]
            selected_similarities = compute_hint_similarities(
                selected_hint.hint_word,
                board,
                hint_embeddings,
                codenames_embeddings,
            )

            # Show hint details
            st.markdown(f"**Cliff size:** {selected_hint.cliff_size:.3f}")
            st.markdown(f"**Covers:** {', '.join(selected_hint.team_words)}")

    st.markdown("---")

    # Display board with optional similarities
    st.subheader("Board")
    display_board(board, user_team, selected_similarities)

    # Show all hints in expanders below
    if hints:
        st.markdown("---")
        st.subheader("All Hints")
        display_hints(hints)


if __name__ == "__main__":
    main()
