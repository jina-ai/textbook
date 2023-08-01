import functools
from importlib import import_module
from typing import Optional, Dict, Type

from typing import Annotated

import torch

from textbook.dataset import DummyDataset
from textbook.model import BaseModule

import transformers
import tempfile

from typer import Typer
import typer
import wandb

app = Typer(pretty_exceptions_enable=False)


config_to_log: Dict = {}


def log_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global config_to_log
        config_to_log = kwargs
        return func(*args, **kwargs)

    return wrapper


@app.command()
@log_args
def train(
    *,
    module: str = "StarCoder",
    epochs: int = 1,
    micro_batch_size: int = 1,
    batch_size: int = 1,
    learning_rate: float = 3e-4,
    output_dir: Optional[str] = None,
    wandb_run_name: str = "",
    use_wandb: bool = False,
    wandb_project: str = "textbook",
    wandb_log_model: Optional[
        bool
    ] = None,  # will be true by default if use_wandb is true
    local_rank: Annotated[int, typer.Option("--local_rank")] = 0,
    deepspeed: Optional[str] = None,
    debug: bool = False,
):
    module_cls: Type[BaseModule] = getattr(import_module("textbook.model"), module)
    module_instance = module_cls(debug=debug)
    model = torch.compile(module_instance.model)
    model = module_instance.model
    tokenizer = module_instance.tokenizer
    dataset = DummyDataset(tokenizer=tokenizer, debug=debug)

    if debug:
        wandb_run_name = "debug"

    if batch_size % micro_batch_size:
        raise ValueError(
            f"batch_size {batch_size} and micro_batch_size {micro_batch_size} are not compatible"
        )

    if wandb_log_model is None:
        wandb_log_model = use_wandb

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
        print(f"temp folder : {output_dir}")

    use_wandb = local_rank == 0 and use_wandb
    if use_wandb:
        run = wandb.init(wandb_project, **dict(config=config_to_log))  # type: ignore

    trainer = transformers.Trainer(
        model=model,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.test_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=batch_size // micro_batch_size,
            optim="adamw_torch",
            warmup_steps=100,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10 if debug else 1,
            save_strategy="epoch" if debug else "no",
            eval_steps=20 if debug else 1,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=False,
            report_to="wandb" if use_wandb else "none",
            run_name=wandb_run_name if use_wandb else None,
            remove_unused_columns=False,
        ),
        data_collator=dataset.data_collator,
    )

    trainer.train()

    if use_wandb and wandb_log_model:
        artifact = wandb.Artifact(name="model_weight", type="model")
        artifact.add_dir(output_dir)
        run.log_artifact(artifact)  # type: ignore


if __name__ == "__main__":
    app()
