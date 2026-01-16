# Codenames Embeddings Explorer

## Project Overview
A tool that generates Codenames hints using OpenAI embeddings and cosine similarity search.

## Architecture

### Embedding Pipeline
1. Words are embedded using OpenAI's text-embedding-3-small model (1536 dimensions)
2. Embeddings are stored as pickle files for quick loading
3. Cosine similarity is computed via numpy dot product on normalized vectors

### Hint Finding Algorithm
1. For each potential hint word in vocabulary:
   - Calculate cosine similarity to all board words
   - Sort board words by similarity
   - Check if top word is a team word
   - Count consecutive team words from top
   - Calculate "cliff" to first non-team word
   - Accept if cliff >= threshold
2. Rank hints by number of team words covered

## Codenames Rules Reference
- Board: 20 words total
- Team distribution: 9 team, 8 enemy, 2 neutral, 1 trap (assassin)
- Hint format: A single word + number (e.g., "BEACH 3")
- Goal: Team guesses all their words based on hints
- Trap: If guessed, instant loss

## Usage

### Setup
```bash
uv sync
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Generate Embeddings
```bash
uv run python Scripts/construct_embeddings.py
```

### Run Streamlit App
```bash
uv run streamlit run app.py
```

## Development Guidelines

### Code Style
- Use type hints for all functions
- Follow PEP 8 conventions
- Use dataclasses for structured data

### File Organization
- `Scripts/` contains standalone scripts
- `Storage/` contains generated data (git-ignored)
- `Data/` contains input word lists

### Environment Variables
- `OPENAI_API_KEY`: Required for embedding generation (system variable or .env file)

## Rules for AI Assistants
- Always use uv for package management (not pip directly)
- Run scripts with `uv run python <script>`
- Never commit `Storage/*.pkl` files
- Keep hint finding algorithm in `Scripts/find_best_word.py`
- Use session state for Streamlit app state management
- Embeddings are normalized - use dot product for cosine similarity
