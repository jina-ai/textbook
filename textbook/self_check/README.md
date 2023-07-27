# Self-checking utility
This module provides utilities to perform self-checking on LLM generated exercises.
The self-checking relies on:
1. self-generating a simple unit test
2. execute the simple unit test using `openai/human-eval` repository

## Setup
Install the `textbook/self_check/requirements.txt` file:
```shell
pip install -r textbook/self_check/requirements.txt
```

You also need to setup an `OPENAI_API_KEY` in the environment variables.

## Usage
You can run an example script from the CLI like so: `python -m textbook.self_check.check_exercise`.
To use it in code, you can do the following:
```python
from textbook.self_check.check_exercise import self_check_problem
prompt = '''
from typing import List
def valid_guessing_letters(word: str, guesses: List[str]) -> List[str]:
    """
    Returns a list of valid guessing letters, which are letters that have not been guessed yet and
    are present in the word.
    Parameters:
    word (str): The word to guess.
    guesses (List[str]): A list of letters that have already been guessed.
    Returns:
    List[str]: A list of valid guessing letters.
    """
'''

completion = '''
    valid_letters = []
    for letter in word:
        if letter not in guesses and letter not in valid_letters:
            valid_letters.append(letter)
    return valid_letters
'''

print(self_check_problem(prompt=prompt, completion=completion))
```
```text
{'passed': True, 'result': 'passed'}
```

