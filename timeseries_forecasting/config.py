from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2


@dataclass(frozen=True)
class WindowConfig:
    seq_lens: List[int] = field(default_factory=lambda: [24, 48, 96, 168])
    pred_lens: List[int] = field(default_factory=lambda: [1, 24, 48])


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    dropout: float = 0.1
    hidden_size: int = 64
    num_layers: int = 2
    patience: int = 10
    weight_decay: float = 0.0
