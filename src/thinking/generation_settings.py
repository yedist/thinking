from dataclasses import dataclass, field


@dataclass
class GenerationSettings:
    max_new_tokens: int = 100
    messages_history: list[dict] = field(default_factory=list)
