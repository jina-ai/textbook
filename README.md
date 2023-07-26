# textbook

reproducing https://arxiv.org/abs/2306.11644


## Install dependency


```cmd
poetry install
poetry shell
pip install torch
```

## run the code


Single gpu run

```cmd
python textbook/train.py --epochs 2 --micro-batch-size 4 --batch-size 128 --learning-rate 1e-4
```

Multiple GPU run



```cmd
deepspeed --num_gpus=2 textbook/train.py --deepspeed ds_config.json --epochs 2 --micro-batch-size 4 --batch-size 128 --learning-rate 1e-4
```

