from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

template = '''
implement a simple test case from the following function. by completing the last example.
The test case is a function that will call the provided function, potentially with some input arguments and should assert the result is the correct output. 
It includes only 1 assert instruction and uses the example from the docstring as test case. It always refers to the function as `candidate`. Just give me the test and nothing else

here are some examples
### function
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False

### test
def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False

### function
def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """
    return number % 1.0

### test
def check(candidate):
    assert candidate(3.5) == 0.5

### function
{func_implementation}
### test
'''

llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
prompt = PromptTemplate(
    input_variables=["func_implementation"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt)


def gen_test(func_implementation: str):
    return chain.run(func_implementation)
