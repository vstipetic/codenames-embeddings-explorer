# Codenames Embeddings Explorer

A tool that generates Codenames hints using OpenAI embeddings and cosine similarity search. Given a game board, it finds hint words that are semantically similar to your team's words while being dissimilar to opponent and neutral words.

## How to Use

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- OpenAI API key

### Setup

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/codenames-embeddings-explorer.git
cd codenames-embeddings-explorer
uv sync
```

2. Configure your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Alternatively, set `OPENAI_API_KEY` as a system environment variable.

3. Add input data files to the `Storage/` folder:
   - `unigram_freq.csv` - Hint vocabulary with columns: `word,count` (sorted by frequency)
   - `codename_words.txt` - List of Codenames board words (one per line)

### Generate Embeddings

Before running the app, generate the embedding files:

```bash
uv run python Scripts/construct_embeddings.py
```

This creates:
- `Storage/hint_embeddings.pkl` - Embeddings for hint vocabulary
- `Storage/codenames_embeddings.pkl` - Embeddings for board words
- `Storage/hint_embeddings_faiss.index` - FAISS index for fast similarity search
- `Storage/hint_words_order.pkl` - Word order mapping for FAISS index

### Run the Application

Launch the Streamlit web interface:

```bash
uv run streamlit run app.py
```

### Using the Interface

1. Click **Generate New Board** to create a random 5x5 Codenames board
2. Select your team color (Red or Blue)
3. Adjust the **Cliff Threshold** to control hint quality (higher = safer hints)
4. Use the **Vocabulary Size** slider to limit hints to more common words
5. View generated hints ranked by number of words covered
6. Select a hint to see similarity scores displayed on the board

## Technical Details

### Architecture

The system uses a three-stage pipeline:

1. **Embedding Generation** (`Scripts/construct_embeddings.py`)
   - Words are embedded using OpenAI's `text-embedding-3-small` model (1536 dimensions)
   - Embeddings are L2-normalized for cosine similarity via dot product
   - A FAISS IndexFlatIP index is built for efficient nearest neighbor search

2. **Hint Finding** (`Scripts/find_best_word.py`)
   - FAISS retrieves candidate hints similar to team words
   - Candidates are filtered (no board word substrings, frequency limits)
   - Vectorized similarity computation via NumPy matrix multiplication
   - Hints are evaluated and ranked by coverage and cliff size

3. **Web Interface** (`app.py`)
   - Streamlit app with session state for board persistence
   - Real-time hint generation with configurable parameters
   - Visual similarity display when hints are selected

### Hint Finding Algorithm

For each candidate hint word:

1. Compute cosine similarity to all 25 board words
2. Sort board words by descending similarity
3. Verify the top word belongs to your team
4. Count consecutive team words from the top
5. Calculate the "cliff" (similarity gap) to the next non-team word
6. If cliff is insufficient, check for internal cliffs between team words
7. Accept the hint with the maximum valid coverage

**Cliff Threshold**: The minimum similarity gap required between the last covered team word and the first non-covered word. Higher thresholds produce safer but potentially lower-coverage hints.

### Filtering Rules

Hints are rejected if they:
- Exactly match any board word
- Contain any board word as a substring (e.g., "BEACHES" when "BEACH" is on the board)
- Fall outside the selected vocabulary size (frequency filtering)

### Board Configuration

Following standard Codenames rules:
- 25 words in a 5x5 grid
- 9 words for the starting team
- 8 words for the opposing team
- 7 neutral words
- 1 assassin (trap) word

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `openai` | Embedding generation via API |
| `numpy` | Vectorized similarity computation |
| `faiss-cpu` | Fast nearest neighbor search |
| `streamlit` | Web interface |
| `python-dotenv` | Environment variable management |

### File Structure

```
codenames-embeddings-explorer/
├── app.py                          # Streamlit web application
├── Scripts/
│   ├── construct_embeddings.py     # Embedding generation script
│   ├── find_best_word.py           # Hint finding algorithm
│   └── validate_data.py            # Input data validation
├── Storage/                        # Data files (git-ignored)
│   ├── unigram_freq.csv            # Hint vocabulary (input)
│   ├── codename_words.txt          # Board words (input)
│   ├── hint_embeddings.pkl         # Generated embeddings
│   ├── codenames_embeddings.pkl    # Generated embeddings
│   ├── hint_embeddings_faiss.index # FAISS index
│   └── hint_words_order.pkl        # Word order mapping
├── pyproject.toml                  # Project configuration
└── .env.example                    # Environment template
```

### Performance Optimizations

- **FAISS indexing**: Reduces candidate set from 100k+ words to ~5000 per team word
- **Vectorized operations**: Matrix multiplication instead of per-word loops
- **Early rejection**: Pre-filters hints where max team similarity < max non-team similarity
- **Frequency filtering**: Optional vocabulary limiting to most common words

## Conclusion

Although the idea is interesting, word embeddings are not optimal for hint generation. Too often, words with similar embeddings are not conventionally similar enough for a good hint, and embeddings can miss subtle associations (antonyms, sayings, or cultural references).  
That said, the hints work fairly often and are rarely nonsensical. Limiting the hint vocabulary improves quality by reducing rare or non-existent words, and tuning the cliff threshold can produce consistently decent hints.