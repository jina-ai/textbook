from textbook.self_check.check_exercise import self_check_problem


def test_check_exercise_passed():
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

    assert self_check_problem(prompt=prompt, completion=completion)['passed']

def test_check_exercise_failed():
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
    return list(set(list1) & set(list2)) + [20]
        """

    assert not self_check_problem(prompt=prompt, completion=completion)['passed']
