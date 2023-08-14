import json
import tempfile
from typing import Optional, Union, List

import torch
from transformers import (
    PreTrainedTokenizer,
    PreTrainedModel,
    StoppingCriteria,
    StoppingCriteriaList,
)
from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

STOP_WORDS = ["\nclass", "\ndef", "\n@", "\nprint", "\nif"]


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, tokenizer, start_length=0):
        self.start_length = start_length
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any([stop_string in decoded_generation for stop_string in STOP_WORDS])
            )
        return all(done)


def _stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


def read_jsonl_file(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_data = json.loads(line)
            data.append(json_data)
    return data


def generate_one_completion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 512,
) -> List[str]:
    inputs = tokenizer(prompt.rstrip(), return_tensors="pt").to("cuda")
    stopping_criteria = StoppingCriteriaList(
        [EndOfFunctionCriteria(tokenizer, start_length=len(inputs["input_ids"][0]))]
    )
    generation_output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        stopping_criteria=stopping_criteria,
        # do_sample=True,
        # temperature=0.2,
        # top_k=0,
        # top_p=0.95
    )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    generation = output[len(prompt) :]
    generation = prompt + _stop_at_stop_token(generation, STOP_WORDS)
    return generation


def evaluate(
    model: Union[torch.nn.Module, PreTrainedModel],
    tokenizer: PreTrainedTokenizer,
    prompt_template: str = "{prompt}",
    eval_file: str = HUMAN_EVAL,
    eval_size: Optional[int] = None,
    max_new_tokens: int = 512,
):
    model.eval()
    problems = read_problems(evalset_file=eval_file)
    eval_size = eval_size or len(list(problems.items()))
    problems = dict(list(problems.items())[:eval_size])

    # since k=1, no need for more samples
    num_samples_per_task = 1
    samples = [
        dict(
            task_id=task_id,
            completion=generate_one_completion(
                model,
                tokenizer,
                prompt_template.format(prompt=problems[task_id]["prompt"]),
                max_new_tokens=max_new_tokens,
            ),
        )
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        write_jsonl(temp_file.name, samples)

        accuracy_results = evaluate_functional_correctness(
            temp_file.name, k=[1], problem_file=eval_file, problems=problems
        )
        sample_results = read_jsonl_file(f"{temp_file.name}_results.jsonl")

    # merge results and problems
    results = {
        item["task_id"]: {**item, **problems[item["task_id"]]}
        for item in sample_results
    }

    return accuracy_results, results
