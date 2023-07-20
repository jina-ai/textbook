from datasets import load_dataset
from transformers import PreTrainedTokenizer


class TinyStoriesDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        debug: bool = False,
    ):
        self.debug = debug

        split = "train[:2%]" if debug else "train"
        dataset = load_dataset("skeskinen/TinyStories-GPT4", split=split)

        if debug:
            dataset = dataset.select(range(10))
        split_dataset = dataset.train_test_split(test_size=0.1)

        self.train_dataset = split_dataset["train"]
        self.test_dataset = split_dataset["test"]

        self.train_dataset.map(self.get_tokenize_fn(tokenizer))
        self.test_dataset.map(self.get_tokenize_fn(tokenizer))
        self.tokenizer = tokenizer

    @staticmethod
    def get_tokenize_fn(tokenizer: PreTrainedTokenizer):
        def tokenize_fn(input):
            return tokenizer(
                input["story"],
                padding=False,
            )

        return tokenize_fn
