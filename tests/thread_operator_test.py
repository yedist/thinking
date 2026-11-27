from pathlib import Path

import thinking as think


def test(model):
    """
    Checks whether a model can generate error-free output using threads.
    """
    settings = think.GenerationSettings()
    settings.messages_history = [{"role": "user", "content": "hello!"}]  # chat history for example

    generation = model.create_generation(settings, think.ThreadOperator)
    generation.run()  # start a thread

    assert type(generation.output) == str  # '.output' needs to wait for the end


if __name__ == '__main__':
    qwen = think.Model.load(
        "Qwen/Qwen3-0.6B",
        cache = Path(__file__).resolve().parent / "temp_cache"
    )
    test(qwen)
