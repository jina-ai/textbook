import functools
from typing import Optional, Dict

from jerboa2.dataset import TinyStoriesDataset

import torch
import transformers
import tempfile
from jerboa2.model.model import StarCoderTest, StarCoderTiny

from typer import Typer

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
    epochs: int = 1,
    micro_batch_size: int = 1,
    batch_size: int = 1,
    learning_rate: float = 3e-4,
    output_dir: Optional[str] = None,
    wandb_run_name: str = "",
    wandb: bool = False,
    wandb_project: str = "tiny_stories",
    debug: bool = False,
):
    model = StarCoderTest() if debug else StarCoderTiny()
    model = torch.compile(model)
    tokenizer = model.get_tokenizer()
    dataset = TinyStoriesDataset(tokenizer=tokenizer, debug=debug)

    if debug:
        wandb_run_name = "debug"

    if batch_size % micro_batch_size:
        raise ValueError(
            f"batch_size {batch_size} and micro_batch_size {micro_batch_size} are not compatible"
        )

    if output_dir is None:
        output_dir = tempfile.mkdtemp()
        print(f"temp folder : {output_dir}")

    if wandb:
        wandb.init(wandb_project, **dict(config=config_to_log))

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
            save_strategy="steps" if debug else "no",
            eval_steps=20 if debug else 1,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=False,
            report_to="wandb" if wandb else "none",
            run_name=wandb_run_name if wandb else None,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    trainer.train()


if __name__ == "__main__":
    app()
