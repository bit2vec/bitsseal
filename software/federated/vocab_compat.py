"""
Compatibility layer for torchtext vocabulary building and tokenization.
Works with both old (0.6.0) and new (0.14+) torchtext APIs,
and provides fallbacks when torchtext is not available.
"""

import re
from collections import Counter

# Try to use torchtext's build_vocab_from_iterator and get_tokenizer
_TORCHTEXT_AVAILABLE = False
try:
    from torchtext.vocab import build_vocab_from_iterator as _build_vocab
    from torchtext.data.utils import get_tokenizer as _get_tokenizer
    import inspect
    _sig = inspect.signature(_build_vocab)
    _has_min_freq = 'min_freq' in _sig.parameters
    _TORCHTEXT_AVAILABLE = True
except Exception:
    _has_min_freq = False


def _basic_english_tokenizer(text):
    """Simple basic_english tokenizer matching torchtext's behavior."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text.split()


def get_tokenizer(name='basic_english'):
    """
    Get a tokenizer function. Uses torchtext if available, otherwise falls back
    to a simple regex-based tokenizer.
    """
    if _TORCHTEXT_AVAILABLE:
        try:
            return _get_tokenizer(name)
        except Exception:
            pass
    return _basic_english_tokenizer


class SimpleVocab:
    """Simple vocabulary that mimics the torchtext Vocab interface."""

    def __init__(self, stoi, itos, default_index=0):
        self._stoi = stoi
        self._itos = itos
        self._default_index = default_index

    def __getitem__(self, token):
        return self._stoi.get(token, self._default_index)

    def __len__(self):
        return len(self._itos)

    def __contains__(self, token):
        return token in self._stoi

    def set_default_index(self, index):
        self._default_index = index

    def get_stoi(self):
        return self._stoi

    def get_itos(self):
        return self._itos

    def lookup_token(self, index):
        return self._itos[index]

    def lookup_indices(self, tokens):
        return [self[t] for t in tokens]


def build_vocab_from_iterator(iterator, min_freq=1, specials=None):
    """
    Build a vocabulary from an iterator of token lists.
    Compatible with both old and new torchtext.

    Args:
        iterator: Iterator yielding lists of tokens
        min_freq: Minimum frequency for a token to be included
        specials: List of special tokens to include at the beginning

    Returns:
        SimpleVocab object with __getitem__, __len__, set_default_index
    """
    if specials is None:
        specials = []

    # Count all tokens
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    # Build itos (index to string) and stoi (string to index)
    itos = list(specials)
    stoi = {s: i for i, s in enumerate(specials)}

    # Add tokens meeting min_freq threshold, sorted by frequency (descending)
    for token, freq in counter.most_common():
        if freq >= min_freq and token not in stoi:
            stoi[token] = len(itos)
            itos.append(token)

    return SimpleVocab(stoi, itos, default_index=0)
