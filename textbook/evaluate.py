from typing import Any, Dict, List

import torch
from transformers import GenerationConfig, PreTrainedTokenizer
from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def generate_one_completion(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_output = model.generate(
        **inputs,
        max_new_tokens = 512,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)

    return output

def evaluate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer,
    prompt_template: str = "{prompt}",
    eval_file: str = HUMAN_EVAL,
) -> List[Dict[str, Any]]:
    model.eval()

    problems = read_problems(evalset_file=eval_file)
    problems = dict(list(problems.items())[0])


    num_samples_per_task = 1
    samples = [
        dict(task_id=task_id, completion=generate_one_completion(model, tokenizer, prompt_template.format(prompt=problems[task_id]["prompt"])))
        for task_id in problems
        for _ in range(num_samples_per_task)
    ]
    write_jsonl("samples.jsonl", samples)
