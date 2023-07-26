from textbook.dataset_gen import OpenAIGenerator
import pytest

@pytest.mark.openai
def test_generation():
    generator = OpenAIGenerator()
    gen = generator.generate("Hello world")
    assert isinstance(gen, str)

def test_generation_mock(mocker):
    generator = OpenAIGenerator()
    mocker.patch("textbook.dataset_gen.OpenAIGenerator.generate", return_value="Hello world WORLDDDDDDDDDDD")
    gen = generator.generate("Hello world")
    assert isinstance(gen, str)
