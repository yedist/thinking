from pathlib import Path

import thinking as think


def test(model_reference, cache = None):
    """
    Charging test with download,
    and charging without download (after a download has been performed).
    """

    think.Model.load(
        model_reference,
        local_only=False,  # allows downloading
        cache=cache,
    )  # verifies that the model exists locally

    think.Model.load(
        model_reference,
        local_only=True,  # offline loading
        cache=cache,
    )  # checks local loading


if __name__ == '__main__':
    test(
        "Qwen/Qwen3-0.6B",  # lightweight language model
        Path(__file__).resolve().parent / "temp_cache"  # checks the use of a specific cache
    )
