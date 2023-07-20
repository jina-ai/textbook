from jerboa2.model.model import StarCoderTest, StarCoderTiny
from typer import Typer

app = Typer(pretty_exceptions_enable=False)


@app.command()
def train(debug: bool = False):
    model = StarCoderTest() if debug else StarCoderTiny()
    _tokenizer = model.get_tokenizers()
