from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase, PreTrainedModel


class Model:
    """
    An object that encapsulates all the components required for using the model.
    """

    @classmethod
    def load(cls, model_referencing: str | Path, local_only: bool = True) -> Model:
        """
        Loads the model components and initializes itself using them.
        Loading can be restricted to local sources only.

        Args:
            model_referencing: A local path to the model, or the
                model's Repository ID on the Hugging Face Hub.

            local_only: When the parameter is disabled, downloading missing files from
                the Hugging Face Hub is enabled.

        Returns:
            Model: The model object after loading.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_referencing,
            local_files_only=local_only,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_referencing,
            local_files_only=local_only,
        )
        return cls(model, tokenizer)

    @classmethod
    def download(cls, repo_id: str, cache: str | Path = None) -> Model:
        """
        Downloads and loads Hugging Face Hub model files using the model's Repository ID.

        Args:
            repo_id: The model's Repository ID on the Hugging Face Hub.

            cache: Path to the directory where model files are stored.
                The default is determined by the `huggingface_hub` library.

        Returns:
            Model: The model object after downloading and loading.
        """
        path = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache
        )
        return cls.load(path)

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer
