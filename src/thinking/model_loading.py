from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedModel


class Model:
    """
    An object that encapsulates all the components required for using the model.
    """

    @classmethod
    def load(
            cls,
            model_referencing: str | Path,
            local_only: bool = True,
            cache: str | Path = None
    ) -> Model:
        """
        Loads the model components and initializes itself using them.
        Loading can be restricted to local sources only.

        Args:
            model_referencing: A local path to the model, or the
                model's Repository ID on the Hugging Face Hub.

            local_only: When the parameter is disabled, downloading missing files from
                the Hugging Face Hub is enabled.

            cache: Path to the directory where model files are stored.
                The default is determined by the `huggingface_hub` library.

        Returns:
            Model: The model object after loading.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_referencing,
            local_files_only=local_only,
            cache_dir=cache,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_referencing,
            local_files_only=local_only,
            cache_dir=cache,
        )
        return cls(model, tokenizer)

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer
