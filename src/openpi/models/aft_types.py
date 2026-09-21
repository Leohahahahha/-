"""Future supervision is a separate pytree, never part of policy observations."""

from typing import Any

from flax import struct


@struct.dataclass
class AFTTargets:
    actions: Any
    shear: Any
    wrench: Any
    action_mask: Any
    sensor_mask: Any

    @classmethod
    def from_dict(cls, data):
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})

    def to_dict(self):
        return {key: getattr(self, key) for key in self.__dataclass_fields__}
