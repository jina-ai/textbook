import pytest
from textbook.model import ReplitBase, ReplitDebug
import torch


@pytest.mark.slow
def test_replit_base():
    ReplitBase()


def test_replit_debug():
    model = ReplitDebug()

    assert model.model.dtype != torch.float32
    assert model.model.dtype == torch.bfloat16
