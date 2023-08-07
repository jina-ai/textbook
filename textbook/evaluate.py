import json
import re
import tempfile
from typing import Optional, Union

import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def read_jsonl_file(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            json_data = json.loads(line)
            data.append(json_data)
    return data


def generate_one_completion(model: pytorch.nn.Module, tokenizer: PretrainedTokenizer, prompt: str) -> List[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_output = model.generate(
        **inputs,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    matches = list(re.finditer(r"\bdef\s+\w+\s*\(", output))
    if len(matches) > 1:
        output = output[: matches[1].span()[0]]

    if not output.startswith(prompt):
        print("output will not be cleaned properly:", output)
    else:
        output = output[len(prompt) :]
    return output


def evaluate(
    model: Union[torch.nn.Module, PreTrainedModel],
    tokenizer: PreTrainedTokenizer,
    prompt_template: str = "{prompt}",
    eval_file: str = HUMAN_EVAL,
    eval_size: Optional[int] = None,
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
