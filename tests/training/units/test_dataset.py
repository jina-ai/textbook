import pytest

from textbook.dataset import DummyDataset, ExerciseDatast
from textbook.model import Replit

from transformers import PreTrainedTokenizer


@pytest.fixture
def tokenizer() -> PreTrainedTokenizer:
    return Replit().tokenizer


def test_tiny_stories(tokenizer):
    DummyDataset(debug=True, tokenizer=tokenizer)


def test_exercises_dataet(tokenizer):
    ExerciseDatast(debug=True, tokenizer=tokenizer)
