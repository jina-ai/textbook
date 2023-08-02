import pytest
from textbook.train import train


@pytest.mark.parametrize("module", ["Replit", "StarCoder"])
@pytest.mark.parametrize("dataset", ["DummySupervisedDataset", "DummyDataset"])
def test_train(module, dataset):
    train(
        module=module,
        dataset=dataset,
        debug=True,
        epochs=1,
        micro_batch_size=1,
        batch_size=1,
        use_wandb=False,
    )
