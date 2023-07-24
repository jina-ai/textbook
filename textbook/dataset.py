from typing import Protocol
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer


class CustomDataset(Protocol):
    train_dataset: Dataset
    eval_dataset: Dataset

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        debug: bool = False,
    ):
        ...

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

        self.train_dataset = self.train_dataset.map(
            self._get_tokenize_fn(tokenizer),
            batched=True,
            num_proc=4,
            remove_columns=self.train_dataset.column_names,
        )

        self.test_dataset = self.test_dataset.map(
            self._get_tokenize_fn(tokenizer),
            batched=True,
            num_proc=4,
            remove_columns=self.test_dataset.column_names,
        )

    @staticmethod
    def _get_tokenize_fn(tokenizer: PreTrainedTokenizer):
        def tokenize_fn(input):
            return tokenizer(
                input["story"],
            )

        return tokenize_fn
