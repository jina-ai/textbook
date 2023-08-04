from textbook import evaluate
from textbook.model import Replit


def test_evaluate(monkeypatch):
    # Define a replacement function to be used in the test
    def mock_generate_one_completion(model, tokenizer, prompt):
        return "\n  return 1"

    # Monkey patch the 'add_numbers' function with the 'mock_add_numbers' function
    monkeypatch.setattr(
        evaluate, "generate_one_completion", mock_generate_one_completion
    )

    replit = Replit(debug=True)
    accuracy_results, results = evaluate.evaluate(
        model=replit.model,
        tokenizer=replit.tokenizer,
        eval_file="human-eval/data/example_problem.jsonl",
    )

    assert accuracy_results["pass@1"] == 1
    assert results["test/0"]["passed"]
    assert results["test/0"]["result"] == "passed"
