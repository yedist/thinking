import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from .generation_settings import GenerationSettings


class Generation:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, settings: GenerationSettings):
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings

    def run(self) -> str:
        """
        black box
        """
        inputs = self._tokenizer.apply_chat_template(  # default args
            self._settings.messages_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self._model.device)

        with torch.inference_mode():
            outputs = self._model.generate(  # default args
                inputs,
                max_new_tokens=self._settings.max_new_tokens
            )

        generated_tokens = outputs[0, inputs.shape[-1]:]

        return self._tokenizer.decode(  # default args
            generated_tokens,
            skip_special_tokens=True
        )
