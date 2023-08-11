from typing import Protocol
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoConfig,
    PreTrainedModel,
    AutoModelForCausalLM,
    GPTBigCodeConfig,
    GPTBigCodeForCausalLM,
)


class BaseModule(Protocol):
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel

    def __init__(self, debug: bool = False):
        ...


class Replit:
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    base_model = "replit/replit-code-v1-3b"

    config = AutoConfig.from_pretrained(
        "replit/replit-code-v1-3b",
        trust_remote_code=True,
        init_device="cuda",
    )

    debug_config = AutoConfig.from_pretrained(
        "replit/replit-code-v1-3b",
        trust_remote_code=True,
        init_device="cuda",
        n_layers=1,
    )

    def __init__(self, debug: bool = False):
        self._init_tokenizer()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            config=self.config if not debug else self.debug_config,
            trust_remote_code=True,
        )

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token


class StarCoder:
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    base_model = "bigcode/starcoderbase-1b"
    config = GPTBigCodeConfig.from_pretrained(
        "bigcode/starcoderbase-1b",
        init_device="cuda",
    )

    debug_config = GPTBigCodeConfig.from_pretrained(
        "bigcode/starcoderbase-1b",
        init_device="cuda",
        n_layer=1,
    )

    def __init__(self, debug: bool = False):
        self._init_tokenizer()
        self.model = GPTBigCodeForCausalLM.from_pretrained(
            self.base_model,
            config=self.config if not debug else self.debug_config,
        )

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
