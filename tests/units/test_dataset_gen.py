from textbook.dataset_gen import OpenAIGenerator, mass_generation

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
    results = mass_generation(prompts, generator, n=2)

    assert len(results) == 4