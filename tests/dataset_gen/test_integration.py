from textbook.dataset_gen.dataset_gen_cli import generate, filter
import os


def test_cli_dataset_gen(tmp_path):
    generate(
        tree_path="textbook/dataset_gen/tree/professions.json",
        leaves_path="textbook/dataset_gen/tree/subsubtopics.json",
        debug=True,
        debug_speed=-1,
        retries=10,
        pool_size=10,
        output_path=tmp_path,
    )

    filter(exo_path=tmp_path, dataset_file=os.path.join(tmp_path, "dataset.jsonl"))

    assert os.path.exists(os.path.join(tmp_path, "dataset.jsonl"))
