import pytest
from textbook.model import Replit, StarCoder
import torch


@pytest.mark.slow
def test_replit_base():
    Replit()


def test_replit_debug():
    model = Replit(debug=True)
    assert model.model.dtype == torch.float32


@pytest.mark.slow
def test_starcoer_base():
    StarCoder()


def test_starcoder_debug():
    model = StarCoder(debug=True)
    assert model.model.dtype == torch.float32
