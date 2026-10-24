"""Provider boundary. Mock outputs are constant and never read reference answers."""

from typing import Protocol


class Reasoner(Protocol):
    def predict(self, prompt: str, *, language: str) -> str: ...


class MockReasoner:
    """Deterministic plumbing check, deliberately not a reasoning model."""

    def predict(self, prompt: str, *, language: str) -> str:
        return "5"
