import json

from textbook.dataset_gen.dataset_gen import (
    OpenAIGenerator,
    load_prompts,
    mass_generation,
    generation,
    MonkeyGenerator,
    write_results_to_jsonl,
    Results,
    Exercice,
)

import pytest


def mock_openai(mocker):
    mocker.patch(
        "textbook.dataset_gen.dataset_gen.OpenAIGenerator.generate",
        return_value=Exercice(
            problem="def f(x,y):", solution="Hello world WORLDDDDDDDDDDD"
        ),
    )


@pytest.mark.openai
def test_generation():
    generator = OpenAIGenerator()
    gen = generator.generate("Hello world")
    assert isinstance(gen, str)


def test_generation_mock(mocker):
    mock_openai(mocker)
    generator = OpenAIGenerator()
    gen = generator.generate("Hello world")
    assert isinstance(gen, Exercice)


def test_mass_generation(mocker):
    mock_openai(mocker)
    generator = OpenAIGenerator()

    prompts = ["Hello world", "Goodbye world"]
    results = mass_generation(prompts, generator)

    assert len(results) == 2


def test_generation_monkey_generator():
    generator = MonkeyGenerator(speed=-1)

    prompts = "Hello world"
    generation(prompts, generator)


def test_mass_generation_monkey_generator():
    generator = MonkeyGenerator(speed=-1)

    prompts = ["Hello world", "Goodbye world"] * 20
    results = mass_generation(prompts, generator)

    assert len(results) == 40


def test_load_prompts():
    prompts = load_prompts("tests/data/prompts_debug.jsonl")
    assert len(prompts) == 5


def test_save_results(tmp_path):
    results = [
        Results(
            prompt="Hello world",
            exercice=Exercice(problem="Hello world WORLDDDDDDDDDDD", solution=""),
        ),
        Results(
            prompt="Goodbye world",
            exercice=Exercice(problem="Goodbye world WORLDDDDDDDDDDD", solution=""),
        ),
    ]
    file = f"{tmp_path}/results.jsonl"
    write_results_to_jsonl(file, results)

    with open(file, "r") as f:
        lines = f.readlines()

    prompts = [Results.parse_obj(json.loads(line)) for line in lines]

    assert len(prompts) == 2
    assert prompts[0].prompt == "Hello world"
    assert prompts[0].exercice.problem == "Hello world WORLDDDDDDDDDDD"
    assert prompts[1].prompt == "Goodbye world"
    assert prompts[1].exercice.problem == "Goodbye world WORLDDDDDDDDDDD"
