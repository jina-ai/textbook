import pandas as pd
import random
import json
import os
from tqdm import tqdm
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


if __name__ == "__main__":
    # Load list of topics
    API_KEY = os.environ["API_PASSWORD"]
    openai.api_key = API_KEY
    TOPICS_PATH = "./topics.csv"
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
    for topic in tqdm(topics_list):
        try:
            subtopics = create_subtopics(topic, 10)
            subtopics_list += subtopics
        except Exception:
            print(f"Failed: {topic}")

    for i, subtopic in tqdm(enumerate(subtopics_list)):
        if i > 10:
            break
        try:
            chapters = create_chapters(subtopic, 5)
            chapters_list += chapters
            if chapters[0][3] == 1:
                leaves_for_combination += chapters

        except Exception:
            print(f"Failed: {subtopic}")

    with open('chapters.jsonl', 'w') as f:
        for item in chapters_list:
            f.write(json.dumps(item) + "\n")
