from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import override


class ForcingFunction(StrEnum):
    DIRAC = "Dirac"
    RICKER = "Ricker"
    DGAUSSIAN = "dGaussian"
    EXTERNAL = "External"


@dataclass(frozen=True)
class SourceConfiguration(ABC):
    x: float
    z: float
    source_surf: bool = False
    vx: float = 0
    vz: float = 0
    angle: float = 0

    forcing_function: ForcingFunction = ForcingFunction.RICKER
    factor: float = 1.0
    tshift: float = 0.0
    f0: float = 1.0

    @abstractmethod
    def to_specfem_parameters(self) -> dict: ...
    def _base_specfem_parameters(self) -> dict:
        return {
            "x": f"{self.x:.5f}",
            "z": f"{self.z:.5f}",
            "source_surf": self.source_surf,
            "angle": self.angle,
            "vx": f"{self.vx:.5f}",
            "vz": f"{self.vz:.5f}",
            str(self.forcing_function): {
                "factor": f"{self.factor:.5g}",
                "tshift": f"{self.tshift:.5g}",
                "f0": f"{self.f0:.5g}",
            },
        }


@dataclass(frozen=True)
class ForceSourceConfiguration(SourceConfiguration):
    @override
    def to_specfem_parameters(self) -> dict:
        if self.forcing_function == ForcingFunction.EXTERNAL:
            e = ValueError(
                "Cannot use `EXTERNAL` forcing function with a `FORCE` "
                "configuration"
            )
            raise e
        return {str(SourceType.FORCE): self._base_specfem_parameters()}


@dataclass(frozen=True)
class CosseratSourceConfiguration(SourceConfiguration):
    @override
    def to_specfem_parameters(self) -> dict:
        if self.forcing_function == ForcingFunction.EXTERNAL:
            e = ValueError(
                "Cannot use `EXTERNAL` forcing function with a `COSSERAT` "
                "configuration"
            )
            raise e
        return {str(SourceType.COSSERAT): self._base_specfem_parameters()}


@dataclass(frozen=True)
class MomentTensorSourceConfiguration(SourceConfiguration):
    @override
    def to_specfem_parameters(self) -> dict:
        if self.forcing_function == ForcingFunction.EXTERNAL:
            e = ValueError(
                "Cannot use `EXTERNAL` forcing function with a `MOMENT_TENSOR` "
                "configuration"
            )
            raise e
        return {str(SourceType.MOMENT_TENSOR): self._base_specfem_parameters()}


@dataclass(frozen=True)
class UserDefinedSourceConfiguration(SourceConfiguration):
    @override
    def to_specfem_parameters(self) -> dict:
        if self.forcing_function != ForcingFunction.EXTERNAL:
            e = ValueError(
                "Must use `EXTERNAL` forcing function with a `USER_DEFINED` "
                "configuration"
            )
            raise e
        return {str(SourceType.USER_DEFINED): self._base_specfem_parameters()}


@dataclass(frozen=True)
class AdjointSourceConfiguration(SourceConfiguration):
    @override
    def to_specfem_parameters(self) -> dict:
        if self.forcing_function == ForcingFunction.EXTERNAL:
            e = ValueError(
                "Cannot use `EXTERNAL` forcing function with an `ADJOINT` "
                "configuration"
            )
            raise e
        return {str(SourceType.ADJOINT): self._base_specfem_parameters()}


class SourceType(StrEnum):
    FORCE = "force"
    COSSERAT = "cosserat-force"
    MOMENT_TENSOR = "moment-tensor"
    USER_DEFINED = "user-defined"
    ADJOINT = "adjoint-source"

    def __call__(self, *args, **kwargs):
        if self == SourceType.FORCE:
            return ForceSourceConfiguration(*args, **kwargs)
        if self == SourceType.COSSERAT:
            return CosseratSourceConfiguration(*args, **kwargs)
        if self == SourceType.MOMENT_TENSOR:
            return MomentTensorSourceConfiguration(*args, **kwargs)
        if self == SourceType.USER_DEFINED:
            return UserDefinedSourceConfiguration(*args, **kwargs)
        if self == SourceType.ADJOINT:
            return AdjointSourceConfiguration(*args, **kwargs)

        e = ValueError("Unknown SourceType")
        raise e
