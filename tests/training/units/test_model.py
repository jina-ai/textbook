import pytest
from textbook.model import ReplitBase, ReplitDebug, StarCoderBase, StarCoderDebug
import torch


@pytest.mark.slow
def test_replit_base():
    ReplitBase()


def test_replit_debug():
    model = ReplitDebug()
    assert model.model.dtype == torch.float32


@pytest.mark.slow
def test_starcoer_base():
    StarCoderBase()


def test_starcoder_debug():
    model = StarCoderDebug()
    assert model.model.dtype == torch.float32
