import pandas as pd
import random
import pickle
import json
import os
from tqdm import tqdm
import numpy as np
from typing import List, Tuple
import openai


def topics_to_list(topics: str) -> List[str]:
    return list(map(lambda x: x.split(". ")[1], topics.split("\n")))


def create_subtopic_query(topic: str, n: int) -> str:
    return f"For a Python textbook give me {n} subtopics of {topic}. Just provide the titles and give no explanation."


def create_subtopics(topic: Tuple[str, int], n) -> List[Tuple[str, str, int]]:
    query = create_subtopic_query(topic, n)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        temperature=1.5,
    )
    topics_list = topics_to_list(completion.choices[0].message["content"])
    result = [(i,) + topic for i in topics_list]
    return result


def create_chapters(
    subtopic: Tuple[str, str, int], n: int
) -> List[Tuple[str, str, str, int]]:
    query = create_subtopic_query(subtopic[0], n)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        temperature=1.5,
    )
    topics_list = topics_to_list(completion.choices[0].message["content"])
    result = [(i,) + subtopic for i in topics_list]
    return result

def create_prompt(pivot_chapter: Tuple[str, str, str, int], n_combinations: str = 5) -> List[str]:

    random.shuffle(leaves_for_combination)
    combination_chapters = []
    for chapter in leaves_for_combination:
        if chapter[1] != pivot_chapter[1] and chapter[2] != pivot_chapter[2]:
            combination_chapters.append(chapter)

        if len(combination_chapters) == n_combinations:
            break

    prompts = []

    for _, j in enumerate(combination_chapters):
        topic1 = pivot_chapter[0],
        topic2 = j[0]
        profession = professions[np.random.randint(0, len(professions))]

        query = f'''
        Create a code completion exercise on the intersection of “{topic1}” and “{topic2}”.  Write it for a {profession}. 

        The exercise must be of the style: 

        ```
        def name(args):

        """Docstring explaining the exercise"""

        python code to solve the exercise
        ```

        NO CLASSES

        MAKE IT VERY DIFFICULT
        '''
        prompts.append(query)
    return prompts

if __name__ == "__main__":
    # Load list of topics
    API_KEY = os.environ["API_PASSWORD"]
    TOPICS_PATH = "../../tests/dataset_gen/topics.csv"
    openai.api_key = API_KEY

    topics = pd.read_csv(TOPICS_PATH)
    topics = topics.fillna(0)
    topics = topics.iloc[:, :3]
    topics.Topic = topics.Topic.str.split(".").str[1]
    topics.Use = topics.Use.astype(int)
    topics.Mixing = topics.Mixing.astype(int)
    topics_df = topics[topics.Use == 1].reset_index(drop=True)
    topics_df = topics_df.drop("Use", axis=1)
    topics_list = list(zip(topics_df.Topic, topics_df.Mixing))

    leaves_for_combination = []
    subtopics_list = []
    chapters_list = []
    for i, topic in tqdm(enumerate(topics_list)):
        if i > 7:
            break
        try:
            subtopics = create_subtopics(topic, 10)
            subtopics_list += subtopics
        except Exception:
            print(f"Failed: {topic}")

    random.shuffle(subtopics_list)
    for i, subtopic in tqdm(enumerate(subtopics_list)):
        if i > 20:
            break
        try:
            chapters = create_chapters(subtopic, 5)
            chapters_list += chapters
            if chapters[0][3] == 1:
                leaves_for_combination += chapters

        except Exception:
            print(f"Failed: {subtopic}")

    random.shuffle(chapters_list)

    with open('./professions', 'rb') as f:
        professions = pickle.load(f)

    with open('chapters.jsonl', 'w') as f:
        for item in chapters_list:
            f.write(json.dumps(item) + "\n")

    for i in range(2):
        pivot = chapters_list[i]
        print("###### PIVOT ##### \n", pivot[0])
        print(create_prompt(pivot))


