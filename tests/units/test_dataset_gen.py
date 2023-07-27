from networkx import rescale_layout
from torch import le
from textbook.dataset_gen.dataset_gen import OpenAIGenerator, load_prompts, mass_generation,generation, MonkeyGenerator

import pytest
from unittest.mock import Mock, patch


def mock_openai(mocker):
    mocker.patch("textbook.dataset_gen.OpenAIGenerator.generate", return_value="Hello world WORLDDDDDDDDDDD")

@pytest.mark.openai
def test_generation():
    generator = OpenAIGenerator()
    gen = generator.generate("Hello world")
    assert isinstance(gen, str)

def test_generation_mock(mocker):
    mock_openai(mocker)
    generator = OpenAIGenerator()
    gen = generator.generate("Hello world")
    assert isinstance(gen, str)



def test_mass_generation(mocker):
    mock_openai(mocker)
    generator = OpenAIGenerator()

    prompts = ["Hello world", "Goodbye world"]
    results = mass_generation(prompts, generator)

    assert len(results) == 2


def test_generation_monkey_generator():

    generator = MonkeyGenerator(speed=-1)

    prompts = ["Hello world", "Goodbye world"] * 20
    results = generation(prompts, generator)


def test_mass_generation_monkey_generator():

    generator = MonkeyGenerator(speed=-1)

    prompts = ["Hello world", "Goodbye world"] * 20
    results = mass_generation(prompts, generator)

    assert len(results) == 40



def test_load_prompts():
    prompts = load_prompts("tests/data/prompts_debug.jsonl")
    assert len(prompts) == 5
    
    
