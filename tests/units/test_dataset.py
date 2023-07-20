import pytest

from jerboa2.dataset import TinyStoriesDataset
from jerboa2.model.model import StarCoderBase

from transformers import PreTrainedTokenizer


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    return StarCoderBase.get_tokenizer()


def test_tiny_stories(tokenizer):
    TinyStoriesDataset(debug=True, tokenizer=tokenizer)
