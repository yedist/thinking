from pathlib import Path

import thinking as think


def test(model):
    """
    Checks whether a model succeeds in generating output without an error.
    """
    settings = think.GenerationSettings()
    settings.messages_history = [{"role": "user", "content": "hello!"}]  # chat history for example
    assert type(model.create_generation(settings).run()) == str


if __name__ == '__main__':
    qwen = think.Model.load(
        "Qwen/Qwen3-0.6B",
        cache = Path(__file__).resolve().parent / "temp_cache"
    )
    test(qwen)
