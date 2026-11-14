"""BEAT public client API. Importing this package does not launch Julia."""

from .backends import BackendInfo, backend_catalog, backend_info
from .client import EngineWorker, WorkerPool
from .paths import EnginePaths, engine_paths

__version__ = "0.8.0"
__all__ = [
    "BackendInfo",
    "backend_catalog",
    "backend_info",
    "EnginePaths",
    "EngineWorker",
    "WorkerPool",
    "engine_paths",
]
