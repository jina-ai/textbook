import json
import os

from textbook.dataset_gen.dataset_gen import (
    OpenAIGenerator,
    load_prompts,
    mass_generation,
    generation,
    MonkeyGenerator,
    write_results_to_jsonl,
    Result,
    generator_to_exercises,
    split_exercises,
    check_exercise,
)
import numpy as np
import pytest


def mock_openai(mocker):
    mocker.patch(
        "textbook.dataset_gen.dataset_gen.OpenAIGenerator.generate",
        return_value=Result(
            prompt="Cheesecake with strawberries",
            output='def gruyere(): """No way jose""" return 0' * 2,
        ),
    )


def update_progress():
    ...


@pytest.mark.openai
def test_generation():
    generator = OpenAIGenerator()
    gen = generator.generate("Hello world")
    assert isinstance(gen, Result)


def test_generation_mock(mocker):
    mock_openai(mocker)
    generator = OpenAIGenerator()
    gen = generator.generate("Hello world")
    prompts = "Hello World"
    generation(prompts, generator, update_progress, 10)
    assert isinstance(gen, Result)
    assert gen.prompt == "Cheesecake with strawberries"
    assert gen.output == 'def gruyere(): """No way jose""" return 0' * 2


def test_mass_generation(mocker, tmp_path):
    mock_openai(mocker)

    def get_generator():
        return OpenAIGenerator()

    prompts = ["Hello world", "Goodbye world"]
    mass_generation(prompts, get_generator, save_dir=str(tmp_path))

    ls = os.listdir(tmp_path)
    assert len(ls) > 0

    file_path = os.listdir(os.path.join(tmp_path, ls[0]))
    assert len(file_path) > 0


def test_generation_monkey_generator():
    n_functions = np.random.randint(0, 100)
    generator = MonkeyGenerator(speed=-1, n_functions=n_functions)
    prompts = "Hello world"
    result = generation(prompts, generator, update_progress, 10)
    assert len(result) == n_functions


def test_mass_generation_monkey_generator(mocker, tmp_path):
    n_functions = np.random.randint(1, 100)

    def get_generator():
        return MonkeyGenerator(speed=-1, n_functions=n_functions)

    prompts = ["Hello world", "Goodbye world"] * 20
    mass_generation(prompts, get_generator, save_dir=str(tmp_path))
    ls = os.listdir(tmp_path)
    assert len(ls) > 0

    file_path = os.listdir(os.path.join(tmp_path, ls[0]))
    assert len(file_path) > 0


def test_load_prompts():
    prompts = load_prompts("tests/data/prompts_debug.jsonl", "prompt")
    assert len(prompts) == 5
    assert isinstance(prompts[0], str)


def test_save_results(tmp_path):
    results = [
        Result(
            prompt="Hello world",
            output='def gruyere(): """No way jose""" return 0',
        ),
        Result(
            prompt="Goodbye world",
            output='def emmentaler(): """No way jose""" return 1',
        ),
    ]
    file = f"{tmp_path}/results.jsonl"
    write_results_to_jsonl(file, results)

    with open(file, "r") as f:
        lines = f.readlines()

    prompts = [Result.parse_obj(json.loads(line)) for line in lines]

    assert len(prompts) == 2
    assert prompts[0].prompt == "Hello world"
    assert prompts[0].output == 'def gruyere(): """No way jose""" return 0'
    assert prompts[1].prompt == "Goodbye world"
    assert prompts[1].output == 'def emmentaler(): """No way jose""" return 1'


def test_split_exercises():
    input = '''
    ```python
    def reverse_name(name: str) -> str:
        """Reverses the letters of a name and returns it.

        >>> reverse_name("LeBron")
        'norBeL'
        >>> reverse_name("Curry")
        'yrruC'
        """
        return name[::-1]

    def reverse_words(sentence: str) -> str:
        """Reverses the order of words in a sentence and returns it.

        >>> reverse_words("I love playing basketball")
        'basketball playing love I'
        >>> reverse_words("Hello World!")
        'World! Hello'
        """
        words = sentence.split()
        return " ".join(words[::-1])

    '''
    assert len(split_exercises(input)) == 2


def test_check_exercise():
    good_exercise = '''
    def cheesecake():
        """Cheesecake is delicious.""""
        return 0
    '''
    another_good_exercise = '''
    def marmelade():
        """Marmelade is delicious.""""
        print("Hello world")
    '''
    bad_exercise = '''
    def blubberfish():
        """Blubberfish is delicious.""""
    '''
    assert check_exercise(good_exercise)
    assert check_exercise(another_good_exercise)
    assert not check_exercise(bad_exercise)


def test_generator_to_functions():
    input = '''
    ```python
    def reverse_name(name: str) -> str:
        """Reverses the letters of a name and returns it.

        >>> reverse_name("LeBron")
        'norBeL'
        >>> reverse_name("Curry")
        'yrruC'
        """
        return name[::-1]

    def reverse_words(sentence: str) -> str:
        """Reverses the order of words in a sentence and returns it.

        >>> reverse_words("I love playing basketball")
        'basketball playing love I'
        >>> reverse_words("Hello World!")
        'World! Hello'
        """
        words = sentence.split()
        return " ".join(words[::-1])

    def reverse_alphabetical_order(names: list) -> list:
        """Reverses the order of names in a list and returns it.

        >>> reverse_alphabetical_order(['LeBron', 'Curry', 'Kobe'])
        ['Kobe', 'Curry', 'LeBron']
        >>> reverse_alphabetical_order(['Jordan', 'Magic', 'Bird'])
        ['Bird', 'Magic', 'Jordan']
        """
        return names[::-1]

    def reverse_phone_number(number: str) -> str:
        """Reverses the order of digits in a phone number and returns it.

        >>> reverse_phone_number("123-456-7890")
        '0987-654-321'
        >>> reverse_phone_number("555-123-4567")
        '7654-321-555'
        """
        area_code, first_half, second_half = number.split("-")
        return second_half + "-" + first_half + "-" + area_code

    def intersection_names_to_frozen_sets(names1: list, names2: list) -> set:
        """Finds the intersection of two lists of names and returns it as a frozen set.

        >>> intersection_names_to_frozen_sets(['LeBron', 'Curry', 'Kobe'], ['Kobe', 'Jordan'])
        {'Kobe'}
        >>> intersection_names_to_frozen_sets(['Bird', 'Magic', 'Jordan'], ['LeBron', 'Kobe', 'Bird'])
        {'Bird'}
        """
        set1 = set(names1)
        set2 = set(names2)
        return frozenset(set1.intersection(set2))
    ```
    '''
    assert len(generator_to_exercises(input)) == 5
