import torch
from transformers import PreTrainedTokenizerBase, PreTrainedModel


class Generation:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, messages: list[dict]):
        self.model = model
        self.tokenizer = tokenizer
        self.messages = messages

    def run(self) -> str:
        """
        black box
        """
        inputs = self.tokenizer.apply_chat_template(  # default args
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(  # default args
                inputs,
                max_new_tokens=200
            )

        generated_tokens = outputs[0, inputs.shape[-1]:]

        return self.tokenizer.decode(  # default args
            generated_tokens,
            skip_special_tokens=True
        )
