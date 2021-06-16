from dataclasses import dataclass


@dataclass()
class HyperParameters:
    BATCH_SIZE: int
    MAX_SENTENCE_LENGTH: int = 1000
