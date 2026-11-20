import thinking as think


def test(model_reference):
    """
    The test does not define checks for the results,
    it only ensures that there are no errors.
    """

    think.Model.load(model_reference, local_only=False)  # verifies that the model exists locally
    think.Model.load(model_reference, local_only=True)  # checks local loading


if __name__ == '__main__':
    test(
        "Qwen/Qwen3-0.6B"  # lightweight language model
    )
