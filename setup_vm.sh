apt update -y
apt install gh screen neovim nvtop -y
pip install -U poetry
gh auth login
gh repo clone jina-ai/textbook
cd textbook
poetry config virtualenvs.create false \
     && poetry install --no-interaction --no-ansi

poetry run pip install torch
