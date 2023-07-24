import pytest
from textbook.model import ReplitBase, ReplitDebug


@pytest.mark.slow
def test_star_coder_init():
    ReplitBase()


def test_star_coder_init():
    ReplitDebug()