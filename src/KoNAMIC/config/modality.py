from enum import Enum


class Modality(str, Enum):
    SENSOR = "sensor"
    VISION = "vision"

    @property
    def key(self) -> str:
        return str(self.value)

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(modality.key for modality in cls)
