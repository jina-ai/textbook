from typing import Protocol
import random

from datasets import Dataset
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


class DummyDataset:
    @staticmethod
    def gen(n: int = 10_000, upper_bound: int = 512):
        for _ in range(n):
            random_integer = random.randint(1, upper_bound)
            yield {"text": "hello world" * random_integer}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        debug: bool = False,
    ):
        self.debug = debug

        dataset = Dataset.from_generator(self.gen)

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
                input["text"],
            )

        return tokenize_fn
