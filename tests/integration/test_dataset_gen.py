from textbook.dataset_gen.dataset_gen_cli import generate

def test_cli_dataset_gen(tmp_path):
    generate(prompt_path="tests/data/prompts_debug.jsonl", debug=True, debug_speed=-1, retries=10, pool_size=10, output_path=tmp_path/"results.jsonl")