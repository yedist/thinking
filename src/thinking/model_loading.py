from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedModel


def _load_model(model_path: str | Path):
    return (
        AutoTokenizer.from_pretrained(model_path),
        AutoModelForCausalLM.from_pretrained(model_path)
    )


class Model:
    @classmethod
    def local_loading(cls, model_path: str | Path):
        return cls(*_load_model(model_path))

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer
