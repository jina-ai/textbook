from human_eval.execution import check_correctness
from textbook.self_check.gen_test import gen_test
import re

IMPORTS = """
from typing import *
"""

def extract_function_name(code_str):
    pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\("
    match = re.search(pattern, code_str)
    if match:
        return match.group(1)
    else:
        return None

def self_check_problem(prompt: str, completion: str):
    test_case = gen_test(IMPORTS + '\n' + prompt + '\n' + completion)
    res = check_correctness(problem={
        'task_id': 'test/0',
        'prompt': prompt,
        'canonical_solution': completion,
        'entry_point': extract_function_name(prompt),
        'test': test_case,
    }, completion=completion, timeout=10)

    return {'passed': res.get('passed'), 'result': res.get('result')}


if __name__ == '__main__':

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
