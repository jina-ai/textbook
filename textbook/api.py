from typer import Typer
import typer
from typing import Annotated

app = Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    local_rank: Annotated[int, typer.Option("--local_rank")] = 0,
):
    print(local_rank)


if __name__ == "__main__":
    app()
