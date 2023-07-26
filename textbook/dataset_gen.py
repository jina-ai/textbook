from concurrent.futures import ThreadPoolExecutor
from math import e, exp
import random
from sre_constants import SUCCESS
import threading
import time

from typing import List, Protocol, Tuple

import openai
from rich.progress import Progress


class Generator(Protocol):

    def generate(self, prompt: str) -> str:
        ...

class OpenAIGenerator():

    def __init__(self, model: str = "gpt-3.5-turbo"):

        self.model = model
    
    def generate(self, prompt: str) -> str:
        chat_completion = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": "Hello world"}])
        return chat_completion.choices[0].message.contentss


class GenerationError(RuntimeError):
    ...

class MonkeyGenerator():
    """
    A generator that failed from time to time and take a random time to respond
    """
    def __init__(self, speed: int = 2):
        self.speed = speed
    
    def generate(self, prompt: str) -> str:
        
        seed = random.randint(0, 100)

        if self.speed > 0:
            time.sleep(seed/100*self.speed)
        if not(seed % 10):
            raise GenerationError("Monkey failed")
        
        return "monkey" * int(seed/10)
    

def generation(prompt: str, generator: Generator, retries: int = 10) -> Tuple[str, str]:

        succes = False
        for i in range(retries):
            try:
                results =  generator.generate(prompt)
                succes = True
            except GenerationError:
                print(f"Generation failed for prompt {prompt}, retrying {i+1}/{retries}")
            else:
                break
        
        if succes:
            return (prompt, results)
        else:
            print(f"Generation failed for prompt {prompt}, skipping")
            return (prompt, "")


def mass_generation(prompts: List[str], generator: Generator, pool_size: int =10, retries: int = 10) -> List[Tuple[str, str]]:

    results = []
    with Progress() as progress:

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            
            task = progress.add_task("[red]Generating...", total=len(prompts))
            futures = []
            for i in range(len(prompts)):  # call API 10 times
                futures.append(executor.submit(generation, prompts[i], generator, retries=retries))
            for future in futures:
                result = future.result()
                progress.update(task, advance=1)
                results.append(result)

    return results


