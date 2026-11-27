from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel

from .generation_settings import GenerationSettings
from .operators.operator_abc import Operator


class Generation:
    """
    An object representing a model's output unit.
    It integrates the model object and the generation settings object
    into the generation function.
    Conceptually, it serves as the connecting point between
    "high-level" objects and the "low-level" layer, which consists of more fundamental libraries.
    """

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizerBase,
            settings: GenerationSettings,
            operator: Operator,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._settings = settings
        self._output = None
        self._operator = operator.create_running_unit(self._generation)

    @property
    def output(self):
        if not self._output:
            self._operator.wait_to_stop()
            self._output = next(self._operator.receive_from_worker(no_wait=True))

        return self._output


    def run(self):
        self._operator.run()

    def _generation(self, send_method: Callable[[Any], None]):
        """
        black box
        """
        inputs = self._tokenizer.apply_chat_template(  # default args
            self._settings.messages_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
            enable_thining=False,
        ).to(self._model.device)

        with torch.inference_mode():
            outputs = self._model.generate(  # default args
                inputs,
                max_new_tokens=self._settings.max_new_tokens
            )

        generated_tokens = outputs[0, inputs.shape[-1]:]

        result = self._tokenizer.decode(  # default args
            generated_tokens,
            skip_special_tokens=True
        )

        send_method(result)
