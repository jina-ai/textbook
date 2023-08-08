import json
import numpy as np
import pytest
from textbook.dataset_gen.dataset_gen import (
    Exercise,
    OpenAIGenerator,
    Result,
    MonkeyGenerator,
    mass_generation,
)
import os


@pytest.mark.openai
@pytest.mark.asyncio
async def test_async_generation():
    generator = OpenAIGenerator()
    gen = await generator.agenerate("Hello world")
    assert isinstance(gen, Result)


@pytest.mark.asyncio
async def test_async_generatio_monkey():
    generator = MonkeyGenerator()
    gen = await generator.agenerate("Hello world")
    assert isinstance(gen, Result)
    assert len(gen.output) > 0


@pytest.mark.asyncio
async def test_mass_generation_monkey_generator(tmp_path):
    n_functions = np.random.randint(1, 100)
    generator = MonkeyGenerator(speed=-1, n_functions=n_functions)

    prompts = ["Hello world", "Goodbye world"] * 5
    await mass_generation(prompts, generator, save_dir=str(tmp_path), batch_size=1)

    assert len(os.listdir(tmp_path)) > 5
    with open(f"{tmp_path}/results_0.jsonl", "r") as f:
        lines = f.readlines()
    assert (
        Exercise.parse_obj(json.loads(lines[0])).problem
        == 'def gorilla(): """Empty function for a gorilla"""'
    )
