from typer import Typer

from textbook.dataset_gen.dataset_gen import Generator, load_prompts, mass_generation, OpenAIGenerator, MonkeyGenerator

app = Typer()


@app.command()
def generate(
    prompt_path: str,
    retries: int = 10,
    pool_size: int = 10,
    debug: bool = False,
    debug_speed: int = 2,
):  
    generator: Generator
    
    if not debug:
        generator = OpenAIGenerator()
    else:
        generator = MonkeyGenerator(speed=debug_speed)
    

    prompts = load_prompts(prompt_path)
    results = mass_generation(prompts, generator, pool_size=pool_size, retries=retries)

    print(results)