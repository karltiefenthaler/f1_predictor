from pathlib import Path
import warnings


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def warn(message: str) -> None:
    warnings.warn(message)
