import pytest
from textbook.model import ReplitBase, ReplitDebug


@pytest.mark.slow
def test_replit_base():
    ReplitBase()


def test_replit_debug():
    ReplitDebug()
