import random
import itertools
import json
import numpy as np
from typer import Typer
from typing import List
from textbook.dataset_gen.dataset_gen import (
    Generator,
    load_leaves,
    mass_generation,
    OpenAIGenerator,
    MonkeyGenerator,
)
import openai
import os

from textbook.dataset_gen.create_prompts import Topic, Query

app = Typer()


def create_prompt_query(topic_1: Topic, topic_2: Topic, profession: str) -> str:
    query = f'''
            Create a code completion exercise on the intersection of “{topic_1.topic}” and “{topic_2.topic}”.  
            Write it for a {profession}. 

            The exercise must be of the style: 

            ```
            def name(args):

            """Docstring explaining the exercise"""

            python code to solve the exercise
            ```

            NO CLASSES

            MAKE IT VERY DIFFICULT
            '''
    query = "\n".join([m.lstrip() for m in query.strip().split("\n")])
    return query


def create_prompts(
    topic: Topic,
    combination_options: List[Topic],
    professions: List[str],
    n: int,
) -> List[Query]:
    random.shuffle(combination_options)
    prompts: List[Query] = []

    for loc_topic in combination_options:
        if len(prompts) == n:
            break

        if loc_topic.mixing and loc_topic.parent != topic.parent:
            profession = professions[np.random.randint(0, len(professions))]
            query = create_prompt_query(topic, loc_topic, profession)
            prompts.append(Query(query=query, topic_1=topic, topic_2=loc_topic))

    return prompts


with open("tree/professions.json", "r") as openfile:
    # Reading from json file
    professions = list(json.load(openfile))


@app.command()
def generate(
    leaves_path: str,
    output_path: str,
    retries: int = 10,
    pool_size: int = 10,
    debug: bool = False,
    debug_speed: int = 2,
    n_combinations: int = 200,
    n_prompts: int = 100,
):
    generator: Generator

    openai.api_key = os.environ["OPENAI_API_KEY"]
    if not debug:
        generator = OpenAIGenerator()
    else:
        generator = MonkeyGenerator(speed=debug_speed)

    leaves = load_leaves(leaves_path)

    prompts: List[List[Query]] = [
        create_prompts(
            i,
            combination_options=leaves,
            professions=professions,
            n=n_combinations,
        )
        for i in leaves
    ]

    prompts_flat = list(itertools.chain(*prompts))
    prompts_selection = [i.query for i in prompts_flat][:n_prompts]

    mass_generation(
        prompts_selection,
        generator,
        save_dir=output_path,
        save_every=int(n_prompts / 10),
        pool_size=pool_size,
        retries=retries,
    )


if __name__ == "__main__":
    app()
