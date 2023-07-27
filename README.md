# textbook

reproducing https://arxiv.org/abs/2306.11644


## Install dependency


```cmd
poetry install
poetry shell
pip install torch
```

## Training  


Single gpu run

```cmd
python textbook/train.py --epochs 2 --micro-batch-size 4 --batch-size 128 --learning-rate 1e-4
```

Multiple GPU run



```cmd
deepspeed --num_gpus=2 textbook/train.py --deepspeed ds_config.json --epochs 2 --micro-batch-size 4 --batch-size 128 --learning-rate 1e-4
```


## setup runpod

bash <(curl -Ls https://raw.githubusercontent.com/jina-ai/textbook/main/setup_vm.sh)


## Generating Dataset

```python 
python textbook/dataset_gen/dataset_gen_cli.py --pool-size 10 "tests/data/prompts_debug.jsonl"
```
