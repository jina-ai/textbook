from textbook.train import train
from datasets import disable_caching

disable_caching()


def test_train():

    train(debug=True, epochs=1, micro_batch_size=1, batch_size=1, wandb=False)
