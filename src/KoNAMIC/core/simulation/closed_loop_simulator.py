from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from KoNAMIC.core.scenarios import Scenario

SimulationOutput = TypeVar("SimulationOutput")


class ClosedLoopSimulator(ABC, Generic[SimulationOutput]):
    @abstractmethod
    def run(self, scenario: Scenario) -> SimulationOutput:
        pass