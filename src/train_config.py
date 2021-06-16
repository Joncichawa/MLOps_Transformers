from dataclasses import dataclass


@dataclass()
class HyperParameters:
    BATCH_SIZE: int = 32
    DROPOUT_RATE: float = 0.3
    MAX_SENTENCE_LENGTH: int = 1000
    HIDDEN_LAYER_SIZE: int = 768
