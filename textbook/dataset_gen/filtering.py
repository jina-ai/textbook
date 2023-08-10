from textbook.dataset_gen.dataset_gen import Exercise
from typing import List, Union
import os
from pathlib import Path


def load_one_file(path: Union[Path, str]) -> List[Exercise]:
    with open(path, "r") as f:
        lines = f.readlines()
    return [Exercise.parse_raw(line) for line in lines]


def load_all_exo(path: Union[Path, str]) -> List[Exercise]:
    if isinstance(path, str):
        path = Path(path)
    exos: List[Exercise] = []
    for sub_dir in os.listdir(path):
        for fn in os.listdir(path / sub_dir):
            exos += load_one_file(path / sub_dir / fn)
    return exos


def filter_bad_exos(
    exos: List[Exercise], carac_to_remove=["??", "___"]
) -> List[Exercise]:
    clean_exos: List[Exercise] = []
    for exo in exos:
        keep = True
        for carac in carac_to_remove:
            if carac in exo.solution:
                keep = False
                break

        if keep:
            clean_exos.append(exo)

    return clean_exos


def remove_extra(exos: List[Exercise], carac_to_split=["# Test", "```"]):
    for exo in exos:
        for carac in carac_to_split:
            exo.solution = exo.solution.split(carac)[0]


def load_and_filter_exos(path: Union[Path, str]) -> List[Exercise]:
    exos = load_all_exo(path)
    print(len(exos))
    clean_exos = filter_bad_exos(exos)
    print(len(clean_exos))

    remove_extra(clean_exos)
    return clean_exos
