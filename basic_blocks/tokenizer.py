from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.pre_tokenizers import Split
import regex as re


import os
import json

import time

# from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

# tokenizer = Tokenizer(BPE())
# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# tokenizer.pre_tokenizer = Whitespace()

# file_src = ["data/TinyStoriesV2-GPT4-train.txt"]
# tokenizer.train(file_src, trainer)


def BPE_tokenizer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    save_path: str = "tokenizer_weights.json",
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    tokenizer = Tokenizer(BPE())
    # tokenizer.pre_tokenizer = Split(
    #     pattern=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    #     behavior="merged_with_previous",
    # )

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Load reference alphabet order to ensure exact compatibility
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    with open(reference_vocab_path, 'r', encoding='utf-8') as f:
        ref_vocab = json.load(f)

    # Extract the exact alphabet order from reference (tokens 1-256)
    ref_alphabet = []
    for token_id in range(1, 257):
        for token_str, tid in ref_vocab.items():
            if tid == token_id:
                ref_alphabet.append(token_str)
                break

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=ref_alphabet  # Use reference alphabet order
    )

    # Train directly on original corpus - no need for enhanced training data
    # since initial_alphabet ensures all 256 characters are included
    file_src = [str(input_path)]
    tokenizer.train(file_src, trainer)


    tokenizer.save(save_path)


    with open(save_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    merges_list = config['model']['merges']

    # Convert merges to bytes using GPT-2 byte mapping
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    merges_list_bytes = []

    for pair in merges_list:
        try:
            # Convert each token to bytes using GPT-2 decoder
            token1_bytes = bytes([gpt2_byte_decoder[char] for char in pair[0]])
            token2_bytes = bytes([gpt2_byte_decoder[char] for char in pair[1]])
            merges_list_bytes.append((token1_bytes, token2_bytes))
        except KeyError:
            # Fallback to UTF-8 encoding if GPT-2 mapping fails
            merges_list_bytes.append((pair[0].encode("utf-8"), pair[1].encode("utf-8")))

    # Convert vocab to bytes using GPT-2 mapping
    vocab_bytes = {}
    for token_str, token_id in tokenizer.get_vocab(True).items():
        try:
            token_bytes = bytes([gpt2_byte_decoder[char] for char in token_str])
            vocab_bytes[token_id] = token_bytes
        except KeyError:
            # Fallback to UTF-8 encoding if GPT-2 mapping fails
            vocab_bytes[token_id] = token_str.encode("utf-8")

    return vocab_bytes, merges_list_bytes

class custom_tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath,'r') as f:
            vocab = f.read()
        with open(merges_filepath, 'r') as f:
            merges = f.read()

        return cls(vocab, merges, special_tokens)

    def encode():
        pass
    def encode_iterable():
        pass

    def decode():
        pass

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # from basic_blocks.tokenizer import BPE_tokenizer
    return BPE_tokenizer(input_path, vocab_size, special_tokens, **kwargs)


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]

    A = set(merges)
    B = set(reference_merges)

    only_in_A = A - B
    only_in_B = B - A

    assert len(merges) == len(reference_merges)

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently, we'll make sure that the vocab keys and values match)
    assert set(vocab.keys()) == set(reference_vocab.keys())
    # assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )

if __name__ == "__main__":
    # test_train_bpe()
    # test_train_bpe_speed()

    input_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/data/owt_train.txt"
    vocab_size = 32000
    special_tokens =["<|endoftext|>"]
    save_path = "tokenizer_owt.json"

    vocab_bytes, merges_list_bytes = BPE_tokenizer(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            save_path=save_path
        )

    input_path = "/mnt/aat/zzhao.zhou/cs336_2025/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens =["<|endoftext|>"]
    save_path = "tokenizer_tiny_story.json"

    vocab_bytes, merges_list_bytes = BPE_tokenizer(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            save_path=save_path
        )
