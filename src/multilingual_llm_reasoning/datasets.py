"""JSONL ingestion; translations share IDs across languages."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Problem:
    id: str
    language: str
    prompt: str
    answer: str


def load_problems(path: Path, languages: tuple[str, ...]) -> list[Problem]:
    problems: list[Problem] = []
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_bytes().splitlines(), 1):
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict) or set(raw) != {"id", "language", "prompt", "answer"}:
                raise ValueError("expected id, language, prompt, answer")
            if any(not isinstance(value, str) or not value.strip() for value in raw.values()):
                raise ValueError("all fields must be nonempty strings")
            problem = Problem(**raw)
            key = (problem.id, problem.language)
            if key in seen:
                raise ValueError(f"duplicate problem-language pair {key}")
            seen.add(key)
            if problem.language in languages:
                problems.append(problem)
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line_number}: {exc}") from exc
    ids_by_language = [{p.id for p in problems if p.language == lang} for lang in languages]
    if not ids_by_language or not ids_by_language[0]:
        raise ValueError("dataset contains no requested problems")
    if any(ids != ids_by_language[0] for ids in ids_by_language):
        raise ValueError("requested languages must have the same nonempty set of problem IDs")
    for problem_id in ids_by_language[0]:
        if len({p.answer for p in problems if p.id == problem_id}) != 1:
            raise ValueError(f"reference answers disagree for {problem_id}")
    return problems
