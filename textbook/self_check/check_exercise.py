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
    test_case = gen_test(IMPORTS + "\n" + prompt + "\n" + completion)
    print(test_case)
    res = check_correctness(
        problem={
            "task_id": "test/0",
            "prompt": prompt,
            "canonical_solution": completion,
            "entry_point": extract_function_name(prompt),
            "test": test_case,
        },
        completion=completion,
        timeout=10,
    )

    return {"passed": res.get("passed"), "result": res.get("result")}


if __name__ == "__main__":
    prompt = '''
from typing import List
def common_elements(list1: List[int], list2: List[int]) -> List[int]:
    """
    Returns a list of common elements between two lists. 
    Parameters:
    list1 (List[int]): The first list.
    list2 (List[int]): The second list.
    Returns:
    List[int]: A list of common elements.
    """
    '''

    completion = """
    return list(set(list1) & set(list2))
    """

    print(self_check_problem(prompt=prompt, completion=completion))
