"""
Streamlit app for Codenames hint generation.

Usage:
    uv run streamlit run app.py
"""

import random
from pathlib import Path

import streamlit as st

from Scripts.find_best_word import (
    BoardWord,
    HintCandidate,
    WordCategory,
    find_best_hints,
    load_embeddings,
)


@st.cache_resource
def load_all_embeddings():
    """Load embeddings (cached across reruns)."""
    hint_path = Path("Storage/hint_embeddings.pkl")
    codenames_path = Path("Storage/codenames_embeddings.pkl")

    if not hint_path.exists() or not codenames_path.exists():
        return None, None

    hint_embeddings = load_embeddings(str(hint_path))
    codenames_embeddings = load_embeddings(str(codenames_path))
    return hint_embeddings, codenames_embeddings


def load_codenames_words() -> list[str]:
    """Load Codenames word list from Data folder."""
    path = Path("Data/codenames_words.txt")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def generate_board(words: list[str], user_team: str) -> list[BoardWord]:
    """
    Generate a random 20-word board.

    Distribution:
    - 9 team words (user's team)
    - 8 enemy words
    - 2 neutral words
    - 1 trap (assassin) word
    """
    if len(words) < 20:
        st.error(f"Not enough words in list ({len(words)}). Need at least 20.")
        return []

    selected = random.sample(words, 20)
    random.shuffle(selected)

    board: list[BoardWord] = []

    # Assign categories
    for i, word in enumerate(selected):
        if i < 9:
            category = WordCategory.TEAM
        elif i < 17:
            category = WordCategory.ENEMY
        elif i < 19:
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


def display_board(board: list[BoardWord], user_team: str):
    """Display the board as a 5x4 grid."""
    cols_per_row = 5

    for row in range(4):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            idx = row * cols_per_row + col_idx
            if idx < len(board):
                bw = board[idx]
                style = get_card_style(bw.category, user_team)
                col.markdown(
                    f'<div style="{style}">{bw.word}</div>',
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
    hint_embeddings, codenames_embeddings = load_all_embeddings()
    codenames_words = load_codenames_words()

    # Check if embeddings exist
    if hint_embeddings is None or codenames_embeddings is None:
        st.error(
            "Embeddings not found! Please run the embedding script first:\n\n"
            "```bash\nuv run python Scripts/construct_embeddings.py\n```"
        )
        return

    if not codenames_words:
        st.error("Codenames word list not found at Data/codenames_words.txt")
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

        st.markdown("---")
        st.markdown("### Legend")
        st.markdown(f"- **{'Red' if user_team == 'Red' else 'Blue'}**: Your team")
        st.markdown(f"- **{'Blue' if user_team == 'Red' else 'Red'}**: Enemy team")
        st.markdown("- **Beige**: Neutral")
        st.markdown("- **Black**: Trap (Assassin)")

    # Main content
    if "board" not in st.session_state:
        st.info("Click 'Generate New Board' to start!")
        return

    board = st.session_state.board

    # Display board
    st.subheader("Board")
    display_board(board, user_team)

    st.markdown("---")

    # Generate and display hints
    st.subheader(f"Best Hints for {user_team} Team")

    with st.spinner("Finding best hints..."):
        hints = find_best_hints(
            board_words=board,
            hint_embeddings=hint_embeddings,
            board_embeddings=codenames_embeddings,
            cliff_threshold=cliff_threshold,
            top_k=10,
        )

    display_hints(hints)


if __name__ == "__main__":
    main()
