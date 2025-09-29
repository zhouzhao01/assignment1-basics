import os
import random
import pickle
from pathlib import Path
from typing import List, Optional


def build_document_index(file_path: str, delimiter: str = "<|endoftext|>") -> List[int]:
    """
    Build an index of document start positions in the file.

    Args:
        file_path: Path to the text file
        delimiter: Document separator token

    Returns:
        List of byte positions where documents start
    """
    positions = []

    with open(file_path, 'r', encoding='utf-8') as f:
        positions.append(0)  # First document starts at beginning

        while True:
            line = f.readline()
            if not line:  # EOF
                break

            if delimiter in line:
                # Record position after this line (start of next document)
                current_pos = f.tell()
                positions.append(current_pos)

    # Remove the last position if it's at EOF (no document there)
    if positions and positions[-1] >= os.path.getsize(file_path):
        positions.pop()

    return positions


def _get_cache_path(file_path: str) -> Path:
    """Get cache file path for document index."""
    file_path_obj = Path(file_path)
    cache_name = f".{file_path_obj.name}_doc_index.pkl"
    return file_path_obj.parent / cache_name


def _load_cached_index(file_path: str) -> Optional[List[int]]:
    """Load cached document index if valid."""
    cache_path = _get_cache_path(file_path)

    if not cache_path.exists():
        return None

    try:
        # Check if cache is newer than source file
        cache_mtime = cache_path.stat().st_mtime
        source_mtime = Path(file_path).stat().st_mtime

        if cache_mtime < source_mtime:
            return None  # Cache is stale

        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    except (OSError, pickle.PickleError):
        return None


def _save_index_cache(file_path: str, index: List[int]) -> None:
    """Save document index to cache file."""
    cache_path = _get_cache_path(file_path)

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(index, f)
    except OSError:
        pass  # Silently fail if can't write cache


def read_document_at_position(file_path: str, position: int, delimiter: str = "<|endoftext|>") -> str:
    """
    Read a single document starting at the given byte position.

    Args:
        file_path: Path to the text file
        position: Byte position where document starts
        delimiter: Document separator token

    Returns:
        Document content as string
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        f.seek(position)

        document_lines = []
        while True:
            line = f.readline()
            if not line:  # EOF
                break

            if delimiter in line:
                # Include content before delimiter, exclude delimiter itself
                before_delimiter = line.split(delimiter)[0]
                if before_delimiter.strip():
                    document_lines.append(before_delimiter)
                break
            else:
                document_lines.append(line)

        return ''.join(document_lines).strip()


def random_sample_documents(
    file_path: str,
    num_samples: int,
    delimiter: str = "<|endoftext|>",
    use_cache: bool = True,
    seed: Optional[int] = None
) -> List[str]:
    """
    Randomly sample documents from a large text file.

    Args:
        file_path: Path to the text file
        num_samples: Number of documents to sample
        delimiter: Document separator token (default: "<|endoftext|>")
        use_cache: Whether to use cached document index (default: True)
        seed: Random seed for reproducible sampling

    Returns:
        List of sampled document strings

    Raises:
        ValueError: If num_samples > number of available documents
        FileNotFoundError: If file_path doesn't exist
    """
    if seed is not None:
        random.seed(seed)

    # Try to load cached index first
    index = None
    if use_cache:
        index = _load_cached_index(file_path)

    # Build index if not cached or cache disabled
    if index is None:
        index = build_document_index(file_path, delimiter)
        if use_cache:
            _save_index_cache(file_path, index)

    total_docs = len(index)
    if num_samples > total_docs:
        raise ValueError(f"Requested {num_samples} samples but only {total_docs} documents available")

    # Randomly sample document indices
    selected_indices = random.sample(range(total_docs), num_samples)

    # Read the selected documents
    documents = []
    for idx in selected_indices:
        position = index[idx]
        doc = read_document_at_position(file_path, position, delimiter)
        documents.append(doc)

    return documents


def get_document_count(file_path: str, delimiter: str = "<|endoftext|>", use_cache: bool = True) -> int:
    """
    Get the total number of documents in the file.

    Args:
        file_path: Path to the text file
        delimiter: Document separator token
        use_cache: Whether to use cached document index

    Returns:
        Number of documents in the file
    """
    # Try to load cached index first
    index = None
    if use_cache:
        index = _load_cached_index(file_path)

    # Build index if not cached
    if index is None:
        index = build_document_index(file_path, delimiter)
        if use_cache:
            _save_index_cache(file_path, index)

    return len(index)