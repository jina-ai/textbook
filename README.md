# Textbook
The goal of this project is to distill ChatGPT's Python coding ability into a smaller model with only 1 billion parameters. Our focus is on training the smaller model to solve coding tasks with natural language descriptions, and we use the [HumanEval](https://github.com/openai/human-eval) benchmark to evaluate our model. While we are aware that that benchmark is far from ideal, we believe that it is a good starting point to demonstrate the success of our approach to model distillation. We have drawn some inspiration from efforts to the results reported in the paper _Textbooks Are All You Need_ [(Gunasekar et al. 2023)](https://doi.org/10.48550/arXiv.2306.11644).

This repository consists of two parts:

* Dataset Generation: The code that we used to generate a \~120 million token dataset of Python programming exercises from ChatGPT 3.5.
* Model Fine-tuning: The code that we used to fine-tune the [Starcoder 1b model](https://github.com/bigcode-project/starcoder) using the generated dataset.

The generated exercises dataset is composed of a diverse set of \~120k Python code exercises (~120m total tokens) generated by ChatGPT 3.5. It follows the format of the [Human Eval benchmark](https://github.com/openai/human-eval): Each training sample is split into a Python function signature with a descriptive docstring, and a solution to the exercise.


## Usage
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/notebook_url](https://colab.research.google.com/drive/1T4IfGfDJ8uxgU8XBPpMZivw_JThzdQim?usp=sharing))

You can download and use the model like so:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
        "jinaai/starcoder-1b-textbook", device_map='auto'
    )

tokenizer = AutoTokenizer.from_pretrained("jinaai/starcoder-1b-textbook")

prompt = '''
def unique(l: list):
    """Return sorted unique elements in a list
    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [0, 2, 3, 5, 9, 123]
    """
'''

inputs = tokenizer(prompt.rstrip(), return_tensors="pt").to("cuda")

generation_output = model.generate(
    **inputs,
    max_new_tokens=128,
    eos_token_id=tokenizer.eos_token_id,
    return_dict_in_generate=True,
)

s = generation_output.sequences[0]
output = tokenizer.decode(s, skip_special_tokens=True)

print(output)
```

```text
def unique(l: list):
    """Return sorted unique elements in a list
    >>> unique([5, 3, 5, 2, 3, 3, 9, 0, 123])
    [0, 2, 3, 5, 9, 123]
    """
    return sorted(set(l))
```

## Synthetic exercise creation

Model distillation is the process of transferring some of the skilled performance of large models on specific classes of tasks to significantly smaller models. The purpose is to get performance comparable to the larger model, but at a fraction of the cost and at a vastly quicker speed. The general outline of this strategy is described (without technical implementation details) in [Textbooks Are All You Need](https://doi.org/10.48550/arXiv.2306.11644).

Key to the distillation process is the creation of synthetic data, generated by the larger AI model, to train the smaller model. We have applied this approach to Python programming tasks and are publishing a summary of our methods here along with the synthetic dataset.

For fuller details and implementation code, see the [related GitHub repository](https://github.com/jina-ai/textbook).

### Diversity

The main problem with model-generated synthetic data is its diversity. If we had constructed this dataset by giving ChatGPT 3.5 the same prompt several hundred thousand times, we would get many very similar, if not functionally identical, results. This would reduce the usefulness of the dataset for training. In principle, one might solve the problem by filtering the results for near duplicates, but this is a non-trivial problem, and even if it could be solved, it would be a wasteful and potentially expensive use of the larger model.

And even then, we could not be sure the examples adequately covered the topic. To solve this problem, we introduced a novel scheme for systematically prompting large language models to produce diverse examples.

### Using a topic tree to build diverse prompts

We constructed a hierarchical model of subjects in Python programming, i.e. a topic tree. First, we manually identified 42 general topic areas in Python knowledge, for example, _data structures_ and _sorting algorithms_. We asked an LLM to propose 10 subtopics for each, and then for each of those 420 fine-grained topics, we asked the LLM to generate 5 even more fine-grained sub-subtopics. This resulted in roughly 2000 very fine-grained topics.

We generated prompts by randomly selecting two of those roughly two thousand topics and combining them:

```
Create a code completion exercise on the intersection of {topic 1} and {topic 2}.
```

To increase randomness and diversity in the results, we also constructed a list of 40 professions, like _economist_, _engineer_, and _social worker_, and added them to the prompt:

```
Create a code completion exercise on the intersection of {topic 1} and {topic 2}.
Write it for a {profession}. 
```

In principle, there are approximately two million possible pairs of topics, and with 40 possible professions, this yields 80 million unique prompts. If the response to each prompt averages 100 tokens, this means our method can generate an 8 billion token synthetic dataset while maintaining a high degree of diversity. The dataset used here is only a small sample of the possible total.


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

