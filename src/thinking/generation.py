import torch

from .model_loading import Model


def generate_response(model: Model, messages: list[dict]) -> str:
    """
    black box
    """
    inputs = model.tokenizer.apply_chat_template(  # default args
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.model.device)

    with torch.inference_mode():
        outputs = model.model.generate(  # default args
            inputs,
            max_new_tokens=200
        )

    generated_tokens = outputs[0, inputs.shape[-1]:]

    return model.tokenizer.decode(  # default args
        generated_tokens,
        skip_special_tokens=True
    )
