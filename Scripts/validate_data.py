"""
Validate input data files before embedding generation.

Usage:
    uv run python Scripts/validate_data.py

Checks:
- File encoding (UTF-8)
- Duplicate words
- Empty/whitespace entries
- Non-ASCII characters
- Control characters
- Excessively long words
"""

import csv
import sys
from pathlib import Path


def validate_hint_vocabulary(filepath: str) -> tuple[list[str], list[str]]:
    """
    Validate hint vocabulary CSV file.

    Args:
        filepath: Path to CSV file with 'word' column

    Returns:
        (valid_words, issues): Clean word list and list of issue descriptions
    """
    issues: list[str] = []
    words: list[str] = []

    # Check file exists
    if not Path(filepath).exists():
        return [], [f"File not found: {filepath}"]

    # Try to read with UTF-8
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if "word" not in (reader.fieldnames or []):
                return [], [f"Missing 'word' column in CSV. Found columns: {reader.fieldnames}"]

            for row_num, row in enumerate(reader, start=2):  # Start at 2 (1 is header)
                word = row.get("word", "").strip().lower()

                if not word:
                    issues.append(f"Row {row_num}: Empty word")
                    continue

                words.append(word)

    except UnicodeDecodeError as e:
        return [], [f"UTF-8 encoding error: {e}"]
    except Exception as e:
        return [], [f"Error reading file: {e}"]

    # Check for duplicates
    seen: set[str] = set()
    duplicates: list[str] = []
    unique_words: list[str] = []

    for word in words:
        if word in seen:
            duplicates.append(word)
        else:
            seen.add(word)
            unique_words.append(word)

    if duplicates:
        # Only report first 10 duplicates
        sample = duplicates[:10]
        issues.append(f"Found {len(duplicates)} duplicate words. Sample: {sample}")

    return unique_words, issues


def validate_codenames_words(filepath: str) -> tuple[list[str], list[str]]:
    """
    Validate codenames word list text file.

    Args:
        filepath: Path to text file with one word per line

    Returns:
        (valid_words, issues): Clean word list and list of issue descriptions
    """
    issues: list[str] = []
    words: list[str] = []

    # Check file exists
    if not Path(filepath).exists():
        return [], [f"File not found: {filepath}"]

    # Try to read with UTF-8
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                word = line.strip().lower()

                if not word:
                    issues.append(f"Line {line_num}: Empty line")
                    continue

                words.append(word)

    except UnicodeDecodeError as e:
        return [], [f"UTF-8 encoding error: {e}"]
    except Exception as e:
        return [], [f"Error reading file: {e}"]

    # Check for duplicates
    seen: set[str] = set()
    duplicates: list[str] = []
    unique_words: list[str] = []

    for word in words:
        if word in seen:
            duplicates.append(word)
        else:
            seen.add(word)
            unique_words.append(word)

    if duplicates:
        issues.append(f"Found {len(duplicates)} duplicate words: {duplicates}")

    return unique_words, issues


def check_encoding_issues(words: list[str], max_length: int = 50) -> tuple[list[str], list[str]]:
    """
    Check for words with problematic characters.

    Args:
        words: List of words to check
        max_length: Maximum acceptable word length

    Returns:
        (clean_words, problematic_words): Words that pass/fail the checks
    """
    clean_words: list[str] = []
    problematic: list[str] = []

    for word in words:
        issues = []

        # Check for non-ASCII characters
        if not word.isascii():
            non_ascii = [c for c in word if ord(c) > 127]
            issues.append(f"non-ASCII chars: {non_ascii[:3]}")

        # Check for control characters (ASCII 0-31, except common whitespace)
        control_chars = [c for c in word if ord(c) < 32 and c not in "\t\n\r"]
        if control_chars:
            issues.append(f"control chars: {[ord(c) for c in control_chars]}")

        # Check for excessive length
        if len(word) > max_length:
            issues.append(f"too long ({len(word)} chars)")

        if issues:
            problematic.append(f"'{word}': {', '.join(issues)}")
        else:
            clean_words.append(word)

    return clean_words, problematic


def main() -> int:
    """
    Run all validations and print summary report.

    Returns:
        0 if all validations pass, 1 if there are warnings
    """
    hint_vocab_path = "Storage/unigram_freq.csv"
    codenames_path = "Storage/codename_words.txt"

    print("=" * 60)
    print("Data Validation Report")
    print("=" * 60)

    has_warnings = False

    # Validate hint vocabulary
    print(f"\n[1/2] Validating hint vocabulary: {hint_vocab_path}")
    hint_words, hint_issues = validate_hint_vocabulary(hint_vocab_path)

    if hint_issues:
        has_warnings = True
        print(f"  Issues found:")
        for issue in hint_issues[:10]:  # Limit output
            print(f"    - {issue}")
        if len(hint_issues) > 10:
            print(f"    ... and {len(hint_issues) - 10} more issues")
    else:
        print("  No structural issues found")

    # Check encoding issues for hint words
    if hint_words:
        clean_hints, problematic_hints = check_encoding_issues(hint_words)
        if problematic_hints:
            has_warnings = True
            print(f"  Encoding issues ({len(problematic_hints)} words):")
            for issue in problematic_hints[:5]:
                print(f"    - {issue}")
            if len(problematic_hints) > 5:
                print(f"    ... and {len(problematic_hints) - 5} more")
            hint_words = clean_hints

    print(f"  Valid words: {len(hint_words):,}")

    # Validate codenames words
    print(f"\n[2/2] Validating codenames words: {codenames_path}")
    codenames_words, codenames_issues = validate_codenames_words(codenames_path)

    if codenames_issues:
        has_warnings = True
        print(f"  Issues found:")
        for issue in codenames_issues:
            print(f"    - {issue}")
    else:
        print("  No structural issues found")

    # Check encoding issues for codenames words
    if codenames_words:
        clean_codenames, problematic_codenames = check_encoding_issues(codenames_words)
        if problematic_codenames:
            has_warnings = True
            print(f"  Encoding issues ({len(problematic_codenames)} words):")
            for issue in problematic_codenames:
                print(f"    - {issue}")
            codenames_words = clean_codenames

    print(f"  Valid words: {len(codenames_words):,}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Hint vocabulary: {len(hint_words):,} valid words")
    print(f"  Codenames words: {len(codenames_words):,} valid words")

    if has_warnings:
        print("\n  Status: WARNINGS - Some issues found (filtered out)")
        return 1
    else:
        print("\n  Status: OK - All validations passed")
        return 0


if __name__ == "__main__":
    sys.exit(main())
