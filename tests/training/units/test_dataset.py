import pytest

from textbook.dataset import DummyDataset
from textbook.model import Replit

from transformers import PreTrainedTokenizer


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    return Replit().tokenizer


def test_tiny_stories(tokenizer):
    DummyDataset(debug=True, tokenizer=tokenizer)
