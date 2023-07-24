import pytest

from textbook.dataset import TinyStoriesDataset
from textbook.model.model import StarCoderBase

from transformers import PreTrainedTokenizer


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    return StarCoderBase.get_tokenizer()


def test_tiny_stories(tokenizer):
    TinyStoriesDataset(debug=True, tokenizer=tokenizer)
