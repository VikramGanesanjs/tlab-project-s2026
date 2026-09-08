"""Benchmark-only, worker-safe timing for classification data pipelines."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict


_STAGES = ("volume_read_s", "volume_preprocessing_s", "volume_augmentation_s")


@dataclass(frozen=True)
class DataPipelineTimingProxy:
    """Pickle-safe proxy passed to DataLoader worker dataset instances."""

    totals: object
    lock: object

    def add(self, **seconds: float) -> None:
        with self.lock:  # type: ignore[union-attr]
            for stage, value in seconds.items():
                if stage not in _STAGES:
                    raise ValueError(f"Unknown data-pipeline timing stage: {stage}")
                self.totals[stage] = float(self.totals[stage]) + float(value)  # type: ignore[index]
            self.totals["samples"] = int(self.totals["samples"]) + 1  # type: ignore[index]


class DataPipelineTimer:
    """Own a shared, resettable accumulator for benchmark-only dataset timings."""

    def __init__(self) -> None:
        self._manager = mp.Manager()
        self._totals = self._manager.dict({stage: 0.0 for stage in _STAGES} | {"samples": 0})
        self._lock = self._manager.Lock()
        self.proxy = DataPipelineTimingProxy(self._totals, self._lock)

    def reset(self) -> None:
        with self._lock:
            for stage in _STAGES:
                self._totals[stage] = 0.0
            self._totals["samples"] = 0

    def snapshot(self) -> Dict[str, float]:
        with self._lock:
            return {stage: float(self._totals[stage]) for stage in _STAGES} | {
                "data_pipeline_samples": float(self._totals["samples"])
            }

    def close(self) -> None:
        self._manager.shutdown()
