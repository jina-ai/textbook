import json

from textbook.dataset_gen.dataset_gen import (
    OpenAIGenerator,
    load_prompts,
    mass_generation,
    generation,
    MonkeyGenerator,
    write_results_to_jsonl,
    Results,
    Exercise,
)

import pytest


def mock_openai(mocker):
    mocker.patch(
        "textbook.dataset_gen.dataset_gen.OpenAIGenerator.generate",
        return_value=Exercise(
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
    assert isinstance(gen, Exercise)


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
    prompts = load_prompts("tests/data/prompts_debug.jsonl", 'prompt')
    assert len(prompts) == 5


def test_save_results(tmp_path):
    results = [
        Results(
            prompt="Hello world",
            exercice=Exercise(problem="Hello world WORLDDDDDDDDDDD", solution=""),
        ),
        Results(
            prompt="Goodbye world",
            exercice=Exercise(problem="Goodbye world WORLDDDDDDDDDDD", solution=""),
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


# def test_generator_to_functions():
#     input = '''
#     ```python
#     def reverse_name(name: str) -> str:
#         """Reverses the letters of a name and returns it.
#
#         >>> reverse_name("LeBron")
#         'norBeL'
#         >>> reverse_name("Curry")
#         'yrruC'
#         """
#         return name[::-1]
#
#     def reverse_words(sentence: str) -> str:
#         """Reverses the order of words in a sentence and returns it.
#
#         >>> reverse_words("I love playing basketball")
#         'basketball playing love I'
#         >>> reverse_words("Hello World!")
#         'World! Hello'
#         """
#         words = sentence.split()
#         return " ".join(words[::-1])
#
#     def reverse_alphabetical_order(names: list) -> list:
#         """Reverses the order of names in a list and returns it.
#
#         >>> reverse_alphabetical_order(['LeBron', 'Curry', 'Kobe'])
#         ['Kobe', 'Curry', 'LeBron']
#         >>> reverse_alphabetical_order(['Jordan', 'Magic', 'Bird'])
#         ['Bird', 'Magic', 'Jordan']
#         """
#         return names[::-1]
#
#     def reverse_phone_number(number: str) -> str:
#         """Reverses the order of digits in a phone number and returns it.
#
#         >>> reverse_phone_number("123-456-7890")
#         '0987-654-321'
#         >>> reverse_phone_number("555-123-4567")
#         '7654-321-555'
#         """
#         area_code, first_half, second_half = number.split("-")
#         return second_half + "-" + first_half + "-" + area_code
#
#     def intersection_names_to_frozen_sets(names1: list, names2: list) -> set:
#         """Finds the intersection of two lists of names and returns it as a frozen set.
#
#         >>> intersection_names_to_frozen_sets(['LeBron', 'Curry', 'Kobe'], ['Kobe', 'Jordan'])
#         {'Kobe'}
#         >>> intersection_names_to_frozen_sets(['Bird', 'Magic', 'Jordan'], ['LeBron', 'Kobe', 'Bird'])
#         {'Bird'}
#         """
#         set1 = set(names1)
#         set2 = set(names2)
#         return frozenset(set1.intersection(set2))
#     ```
#     '''
#     assert len(generator_to_exercises(input)) == 6
