from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, T] = {}

    def register(self, key: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            if key in self._items:
                raise KeyError(f"{key!r} is already registered in {self.name}")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, key: str) -> T:
        if key not in self._items:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise KeyError(f"Unknown {self.name}: {key}. Available: {available}")
        return self._items[key]

    def keys(self) -> list[str]:
        return sorted(self._items)
