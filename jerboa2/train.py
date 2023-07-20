from jerboa2.model.model import StarCoderTest, StarCoderTiny
from jerboa2.dataset import TinyStoriesDataset
from typer import Typer

app = Typer(pretty_exceptions_enable=False)


@app.command()
def train(debug: bool = False):
    model = StarCoderTest() if debug else StarCoderTiny()
    tokenizer = model.get_tokenizer()
    _dataset = TinyStoriesDataset(tokenizer=tokenizer, debug=debug)


if __name__ == "__main__":
    app()
