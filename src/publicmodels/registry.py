"""Adapter lookup. Adding a public model means adding one entry here."""
from __future__ import annotations

from src.publicmodels.anticipatory import AnticipatoryAdapter
from src.publicmodels.mmt import MMTAdapter
from src.publicmodels.base import PublicModelAdapter

ADAPTERS: dict[str, type[PublicModelAdapter]] = {
    AnticipatoryAdapter.name: AnticipatoryAdapter,
    MMTAdapter.name: MMTAdapter,
}


def get_adapter(name: str) -> PublicModelAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(ADAPTERS)}")
    return ADAPTERS[name]()
