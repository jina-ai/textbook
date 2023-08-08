import pytest
from textbook.dataset_gen.dataset_gen import OpenAIGenerator, Result, MonkeyGenerator


@pytest.mark.openai
@pytest.mark.asyncio
async def test_async_generation():
    generator = OpenAIGenerator()
    gen = await generator.agenerate("Hello world")
    assert isinstance(gen, Result)


@pytest.mark.asyncio
async def test_async_generatio_monkey():
    generator = MonkeyGenerator()
    gen = await generator.agenerate("Hello world")
    assert isinstance(gen, Result)
    assert len(gen.output) > 0
