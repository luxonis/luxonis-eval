from collections.abc import Iterator
from typing import Any


class LimitedLoader:
    """Bound the visible sample count of an existing loader."""

    def __init__(self, base_loader: Any, max_samples: int) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be a positive integer.")

        self.base_loader = base_loader
        self.max_samples = min(max_samples, len(base_loader))

    def __len__(self) -> int:
        return self.max_samples

    def __iter__(self) -> Iterator[Any]:
        for idx in range(self.max_samples):
            yield self.base_loader[idx]

    def __getitem__(self, idx: int) -> Any:
        if idx < 0:
            idx += self.max_samples
        if idx < 0 or idx >= self.max_samples:
            raise IndexError("LimitedLoader index out of range.")
        return self.base_loader[idx]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_loader, name)
