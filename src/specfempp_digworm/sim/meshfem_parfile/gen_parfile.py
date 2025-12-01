from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import override

from .external_mesh_config import ExternalMesherFileConfig


@dataclass(frozen=True)
class ReceiverSeries:
    nrec: int
    xdeb: float
    zdeb: float
    xfin: float
    zfin: float
    record_at_surface_same_vertical: bool = False

    def __str__(self):
        return f"""
nrec                            = {self.nrec}
xdeb                            = {self.xdeb:f}
zdeb                            = {self.zdeb:f}
xfin                            = {self.xfin:f}
zfin                            = {self.zfin:f}
record_at_surface_same_vertical = .{str(self.record_at_surface_same_vertical).lower()}.
      """


@dataclass(frozen=True)
class MaterialModel(ABC):
    @abstractmethod
    def material_string(self, model_number: int) -> str: ...


@dataclass(frozen=True)
class MaterialModelAcousticElastic(MaterialModel):
    typecode: int = field(init=False, default=1)
    rho: float
    vp: float
    vs: float
    Qkappa: float = 9999
    Qmu: float = 9999

    @override
    def material_string(self, model_number: int) -> str:
        return (
            f"{model_number} 1 {self.rho} {self.vp} {self.vs} 0 0 "
            f"{self.Qkappa} {self.Qmu} 0 0 0 0 0 0"
        )


def bool_to_fortstr(val: bool):
    return f".{str(val).lower()}."


@dataclass
class InternalMesherConfig:
    @dataclass(frozen=True)
    class Region:
        xlow: int
        xhigh: int
        ylow: int
        yhigh: int
        material_index: int

        def __str__(self):
            return (
                f"{self.xlow} {self.xhigh} {self.ylow} "
                f"{self.yhigh} {self.material_index}"
            )

    topography_file: str | PathLike
    xmin: float
    xmax: float
    nx: int
    do_stacey_absorbing: bool
    absorbing_bottom: bool
    absorbing_right: bool
    absorbing_top: bool
    absorbing_left: bool
    model_regions: Collection[Region]

    def param_string(self):
        return f"""
interfacesfile                  = {self.topography_file:s}
xmin                            = {self.xmin}
xmax                            = {self.xmax}
nx                              = {self.nx}
STACEY_ABSORBING_CONDITIONS     = {bool_to_fortstr(self.do_stacey_absorbing)}
absorbbottom                    = {bool_to_fortstr(self.absorbing_bottom)}
absorbright                     = {bool_to_fortstr(self.absorbing_right)}
absorbtop                       = {bool_to_fortstr(self.absorbing_top)}
absorbleft                      = {bool_to_fortstr(self.absorbing_left)}
nbregions                       = {len(self.model_regions)}
{"\n".join(str(reg) for reg in self.model_regions)}
"""


@dataclass
class MeshConfiguration:
    title: str
    nproc: int
    receivers: Collection[ReceiverSeries]
    materials: Collection[MaterialModel]
    output_folder: str | PathLike
    database_output_file: str | PathLike
    stations_filename: str | PathLike
    tomography_file: str | PathLike
    external_mesher_files: ExternalMesherFileConfig | None
    internal_mesher_config: InternalMesherConfig

    def export(self, config_file: str | PathLike):
        exteral_mesh = (
            self.external_mesher_files
            if self.external_mesher_files is not None
            else ExternalMesherFileConfig(Path())
        )

        with Path(config_file).open("w") as f:
            f.write(f"""
title                           = {self.title}
NPROC                           = {self.nproc}\n
OUTPUT_FILES                   = {self.output_folder:s}
database_filename               = {self.database_output_file:s}

PARTITIONING_TYPE               = 3
NGNOD                           = 9
use_existing_STATIONS           = .false.
nreceiversets                   = {len(self.receivers)}
anglerec                        = 0.d0
rec_normal_to_surface           = .false.
{"".join(str(rec) for rec in self.receivers)}
stations_filename              = {self.stations_filename:s}
nbmodels                        = {len(self.materials)}
{"\n".join(model.material_string(i + 1) for i, model in enumerate(self.materials))}
TOMOGRAPHY_FILE                 = {self.tomography_file}
read_external_mesh              = {bool_to_fortstr(self.external_mesher_files is not None)}
{exteral_mesh.param_string()}
{self.internal_mesher_config.param_string()}
output_grid_Gnuplot             = .false.
output_grid_ASCII               = .false.
            """)
