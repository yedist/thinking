from dataclasses import dataclass, field


@dataclass
class GenerationSettings:
    """
    An object that groups settings for generating model output.
    """

    max_new_tokens: int = 100
    messages_history: list[dict] = field(default_factory=list)  # = []
