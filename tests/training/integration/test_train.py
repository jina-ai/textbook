import pytest
from textbook.train import train


@pytest.mark.parametrize("module", ["Replit", "StarCoder"])
def test_train(module):
    train(
        module=module,
        debug=True,
        epochs=1,
        micro_batch_size=1,
        batch_size=1,
        use_wandb=False,
    )
