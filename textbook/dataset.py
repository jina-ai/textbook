from typing import Protocol, Optional
import random

from datasets import Dataset, load_dataset
from transformers import (
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from transformers.data.data_collator import DataCollatorMixin


class CustomDataset(Protocol):
    train_dataset: Dataset
    test_dataset: Dataset
    data_collator: DataCollatorMixin

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        debug: bool = False,
        dataset_name: Optional[str] = None,
    ):
        ...


class DummyDataset:
    @staticmethod
    def gen(n: int = 100_000, upper_bound: int = 512):
        for _ in range(n):
            random_integer = random.randint(1, upper_bound)
            yield {"text": "hello world" * random_integer}

    def __init__(self, tokenizer: PreTrainedTokenizer, debug: bool = False, **kwargs):
        self.debug = debug

        dataset = Dataset.from_generator(self.gen)

        if debug:
            dataset = dataset.select(range(10))

        split_dataset = dataset.train_test_split(test_size=0.1)

        self.train_dataset = split_dataset["train"]
        self.test_dataset = split_dataset["test"]

        self.train_dataset = self.train_dataset.map(
            self._get_preprocess_fn(tokenizer),
            batched=True,
            num_proc=4,
            remove_columns=self.train_dataset.column_names,
        )

        self.test_dataset = self.test_dataset.map(
            self._get_preprocess_fn(tokenizer),
            batched=True,
            num_proc=4,
            remove_columns=self.test_dataset.column_names,
        )

        self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    @staticmethod
    def _get_preprocess_fn(tokenizer: PreTrainedTokenizer):
        def tokenize_fn(input):
            return tokenizer(
                input["text"],
            )

        return tokenize_fn


class ExerciseDatast:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str = "jinaai/code_exercises_40k",
        debug: bool = False,
    ):
        self.debug = debug

        dataset = load_dataset(dataset_name)["train"]

        if debug:
            dataset = dataset.select(range(10))

        split_dataset = dataset.train_test_split(test_size=0.1)

        self.train_dataset = split_dataset["train"]
        self.test_dataset = split_dataset["test"]

        self.train_dataset = self.train_dataset.map(
            self._get_preprocess_fn(tokenizer),
            batched=False,
            num_proc=4,
            remove_columns=self.train_dataset.column_names,
        )

        self.test_dataset = self.test_dataset.map(
            self._get_preprocess_fn(tokenizer),
            batched=False,
            num_proc=4,
            remove_columns=self.test_dataset.column_names,
        )

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

    @staticmethod
    def _get_preprocess_fn(tokenizer: PreTrainedTokenizer):
        def tokenize_fn(input):
            input_problem = input["problem"]
            input_solution = input["solution"]

            inputs = tokenizer(input_problem)
            targets = tokenizer(input_solution)
            inputs["labels"] = [-100] * len(inputs["input_ids"]) + targets[
                "input_ids"
            ]  # we don't train on the problem tokens
            inputs["input_ids"] = inputs["input_ids"] + targets["input_ids"]
            inputs["attention_mask"] = (
                inputs["attention_mask"] + targets["attention_mask"]
            )

            return inputs

        return tokenize_fn
