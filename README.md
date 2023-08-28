# textbook

The goal of this project is to distill chatGPT coding ability into a smaller model down to 1b parameters. We focus on solving the coding exercises task and use the [HumanEval](https://github.com/openai/human-eval) benchmark to evaluate our model. While we are aware that benchmark is far from
being perfect, we believe that it is a good starting point to prove our approach of distilling knowledge from a large model into a smaller one. Part of the idea were inspired by the [textbook](reproducing https://arxiv.org/abs/2306.11644) paper.


The repo consist of two part:


* Dataset generation: We generate a 200 Millions tokens dataset from chatGPT 3.5
* Model finetuning: We finetune starcoder 1b on the generated dataset



## Synthetic exercise creation

Distillation of LLM's describes the process of capturing parts of foundational language model in a significantly smaller model. This allows for similar performance at a fraction of the cost and at vastly quicker speed. In the “textbooks are all you need”(https://arxiv.org/abs/2306.11644) paper this approach is explained for a storytelling LLM. Sadly, the technical details of distillation are not explained. Key to distillation is the creation of the synthetic data, on which the smaller model is trained. We applied the distillation approach to a coding task and more importantly publish our approach on creating a synthetic dataset. This document explains how we created the 40M tokens of synthetic exercises. 

### Diversity

The main problem of any large synthetic dataset is its diversity. Repeatedly (~400.000x) asking a LLM for a Python exercise will result in high similarity between the results. One could filter the exercises afterwards, however, this would increase the cost as you have to create more exercises. Also, it is unclear whether this approach will create exercises that cover all areas of a topic. Therefore, a different method is required to ensure diversity. 

### Knowledge tree

How do we force a LLM to create exercises for different topics? By regarding Python knowledge as a tree structure we can create different subtopics of the broader topic. These subtopics are then used to create exercises. First we curated a list of 42 subtopics of Python. These are topics as “Data Structures” and “Sorting Algorithms”. For each of those topics we created 10 subtopics using a LLM. These subtopics are then split into 5 topics each again, leaving us with ~2000 topics. Assuming 100 tokens per exercise we would have 2000*100 = 200.000  tokens. This is a far cry from the required millions of tokens necessary for knowledge injection during fine-tuning. We therefore combine topics with each other. Each individual topic is combined with 200 other topics to create new, unique topics. In our experiments these combined topics ensure data diversity. By combining topics we can create 200.000 * 200 = 40M tokens of exercises. 

Another way to inject diversity is prompt engineering. By having random aspects in the prompt the LLM is primed to create different exercises. We created a list of 50 professions of which the LLM chose one randomly per exercise. For example: “Write this exercise for a baker” or “Write this exercise for a doctor”. By priming the model with different exercises, different types of exercises are created. A baker might be more associated with baking which is associated with creating objects (bread), whereas a doctor is associated with changing states of patients. Therefore different professions require different exercises. For each of the 200 combinations we randomly selected a profession. If one wanted to create more exercises, you could take the same combination and sample more professions. For example, by using 5 different professions per combination you can create 200M tokens, whilst maintaining diversity.



## Install dependency


```cmd
poetry install
poetry shell
pip install torch
```


## Generating Dataset


Follow this step to reproduce the dataset generation


First export your openAI key 
```shell
export OPENAI_API_KEY=sk-XXX
```
then start to parrallel call to open ai
```shell
cd textbook/dataset_gen
python dataset_gen_cli.py generate ./tree/professions.json ./tree/subsubtopics.json ./exercises --n-prompts 2_000_000 --pool-size 40 
```

this should take around 6hours. The process might be killed before the end but the data will still be save progressivly.


Once the file are generated you can postprocess the files and save it into a jsonl file

```shell 
python dataset_gen_cli.py filter ./exercises dataset.jsonl
```

push to hf dataset

```shell
python dataset_gen_cli.py push "jinaai/code_exercises_40k" dataset.jsonl
```

## Training  


Single gpu run

```cmd
python textbook/train.py --epochs 2 --micro-batch-size 4 --batch-size 128 --learning-rate 1e-4
```

a100 run :


```cmd
python textbook/train.py --module StarCoder --dataset ExerciseDatast --epochs 1 --micro-batch-size 8 --batch-size 128 --wandb-project textbook_debug --use-wandb --no-wandb-log-model
```


```cmd
deepspeed --num_gpus=2 textbook/train.py --deepspeed ds_config.json --epochs 2 --micro-batch-size 4 --batch-size 128 --learning-rate 1e-4
```


Note:

to use starcoder base model you need to first login to HF and accept the ToS of the used starcoder base model (https://huggingface.co/bigcode/starcoderbase-1b)
```cmd
huggingface-cli login
```


## setup runpod

bash <(curl -Ls https://raw.githubusercontent.com/jina-ai/textbook/main/setup_vm.sh)

