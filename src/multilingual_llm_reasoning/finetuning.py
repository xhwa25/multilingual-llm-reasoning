"""LoRA trainer boundary; training dependencies are intentionally not installed."""

from pathlib import Path
from typing import Protocol

from .config import LoRAConfig


class LoRATrainer(Protocol):
    def train(self, dataset: Path, *, model: str, config: LoRAConfig, output: Path) -> None: ...
