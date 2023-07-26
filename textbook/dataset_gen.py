from math import e, exp
import random
from sre_constants import SUCCESS
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
    
    
def mass_generation(prompts: List[str], generator: Generator, n: int = 1, retries: int = 10) -> List[Tuple[str, str]]:

    return_results: List[Tuple[str, str]] = []
    with Progress() as progress:
        task = progress.add_task("[red]Generating...", total=len(prompts)*n)
        for prompt in prompts:
            
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
                return_results.append( (prompt, results))
            else:
                print(f"Generation failed for prompt {prompt}, skipping")

            progress.advance(task)

    return return_results