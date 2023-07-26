import re
from typing import List, Tuple

import openai
from rich.progress import Progress
class OpenAIGenerator():

    def __init__(self, model: str = "gpt-3.5-turbo"):

        self.model = model
    
    def generate(self, prompt: str) -> str:
        chat_completion = openai.ChatCompletion.create(model=self.model, messages=[{"role": "user", "content": "Hello world"}])
        return chat_completion.choices[0].message.contentss
    


def mass_generation(prompts: List[str], generator: OpenAIGenerator, n: int = 1) -> List[Tuple[str, str]]:

    return_results: List[Tuple[str, str]] = []
    with Progress() as progress:
        task = progress.add_task("[red]Generating...", total=len(prompts)*n)
        for _ in range(n):
            for prompt in prompts:
                return_results.append( (prompt, generator.generate(prompt)))
                progress.advance(task)

    return return_results