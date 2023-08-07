from concurrent.futures import ThreadPoolExecutor
import json
import random
import time

from typing import List, Protocol

import openai
from rich.progress import Progress

from pydantic import BaseModel


class Exercise(BaseModel):
    problem: str
    solution: str


class Result(BaseModel):
    prompt: str
    output: str


def split_exercises(output: str) -> List[str]:
    """Split the result of the generation into separate functions"""
    return ["def" + i for i in output.split("def")[1:]]


def check_exercise(exercise: str) -> bool:
    try:
        if (
            "return" not in exercise.split('"""')[2]
            and "print" not in exercise.split('"""')[2]
        ):
            return False
        else:
            return True
    except IndexError:
        return False


def generator_to_exercises(output: str) -> List[Exercise]:
    exercises = split_exercises(output)
    exercises = [i for i in exercises if check_exercise(i)]
    results = []
    for j in exercises:
        try:
            splitted_exercise = j.split('"""')
            question = '"""'.join(splitted_exercise[:2]) + '"""'
            answer = splitted_exercise[2]
            results.append(Exercise(problem=question, solution=answer))
        except IndexError:
            splitted_exercise = j.split("'''")
            question = "'''".join(splitted_exercise[:2]) + "'''"
            answer = splitted_exercise[2]
            results.append(Exercise(problem=question, solution=answer))

    return results


class Generator(Protocol):
    def generate(self, prompt: str) -> Result:
        ...


class OpenAIGenerator:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
    ):
        self.model = model

    def generate(self, prompt: str) -> Result:
        chat_completion = openai.ChatCompletion.create(
            model=self.model, messages=[{"role": "user", "content": prompt}]
        )
        result = Result(
            prompt=prompt, output=chat_completion.choices[0].message.content
        )

        return result


class GenerationError(Exception):
    ...


class MonkeyGenerator:
    """
    A generator with a random response time and a random failure rate
    """

    def __init__(self, speed: int = 2, n_functions: int = 10):
        self.speed = speed
        self.n_functions = n_functions

    def generate(self, prompt: str) -> Result:
        seed = random.randint(0, 100)

        if self.speed > 0:
            time.sleep(seed / 100 * self.speed)
        if not (seed % 10):
            raise GenerationError("Monkey failed")

        return Result(
            prompt=prompt,
            output='def gorilla(): """Empty function for a gorilla""" return 0'
            * self.n_functions,
        )


def generation(
    prompt: str,
    generator: Generator,
    retries: int = 10,
) -> List[Exercise]:
    success = False
    for i in range(retries):
        try:
            result = generator.generate(prompt)
            success = True
        except GenerationError:
            print(f"Generation failed for prompt {prompt}, retrying {i + 1}/{retries}")
        else:
            break

    if success:
        exercises = generator_to_exercises(result.output)
        return exercises

    else:
        print(f"Generation failed for prompt {prompt}, skipping")
        return [Exercise(problem=prompt, solution="")]


def mass_generation(
    prompts: List[str],
    generator: Generator,
    save_dir: str,
    save_every: int,
    pool_size: int = 10,
    retries: int = 10,
) -> List[Exercise]:
    """
    Generate from a list of prompts. Use a thread pool to parallelize the generation with catch and retry mechanism
    """
    results = []
    counter = 0
    with Progress() as progress:
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            task = progress.add_task("[red]Generating...", total=len(prompts))
            futures = []
            for i in range(len(prompts)):  # call API 10 times
                futures.append(
                    executor.submit(generation, prompts[i], generator, retries=retries)
                )
            for future in futures:
                result = future.result()
                progress.update(task, advance=1)
                results += result
                if len(results) == save_every:
                    write_results_to_jsonl(
                        f"{save_dir}/results_{counter}.jsonl", results
                    )
                    results = []
                    counter += 1

    return results


def load_prompts(file: str, key_prompt: str = "prompt") -> List[str]:
    with open(file, "r") as f:
        lines = f.readlines()

    prompts = [json.loads(line)[key_prompt] for line in lines]
    return prompts


def write_results_to_jsonl(file_path: str, results: List[Exercise]):
    with open(file_path, "w") as file:
        for item in results:
            json.dump(item.dict(), file)
            file.write("\n")
