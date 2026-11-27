from pathlib import Path

import thinking as think


def test(model):
    """
    Checks whether a model succeeds in generating output without an error.
    """
    settings = think.GenerationSettings()
    settings.messages_history = [{"role": "user", "content": "hello!"}]  # chat history for example

    generation = model.create_generation(settings, think.SyncOperator)
    generation.run()  # not parallel

    assert type(generation.output) == str


if __name__ == '__main__':
    qwen = think.Model.load(
        "Qwen/Qwen3-0.6B",
        cache = Path(__file__).resolve().parent / "temp_cache"
    )
    test(qwen)
