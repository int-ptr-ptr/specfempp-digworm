from dataclasses import dataclass
from enum import StrEnum
from os import PathLike
from pathlib import Path

from yaml import Dumper, dump

from ..source import SourceConfiguration


class PresetQuadratureRule(StrEnum):
    GLL4 = "GLL4"


class SeismogramTypes(StrEnum):
    DISPLACEMENT = "displacement"
    PRESSURE = "pressure"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


@dataclass
class SiesmogramConfiguration:
    seismo_types: list[SeismogramTypes]
    output_folder: str = "OUTPUT_FILES/seismograms"
    stations_input_file: str = "STATIONS"
    receiver_angle: float = 0.0
    steps_between_samples: int = 1


@dataclass
class DisplayConfiguration:
    steps_between_display: int
    displayed_field: str = "displacement"
    simfield: str = "forward"
    output_format: str = "PNG"
    output_folder: str = "OUTPUT_FILES/display"

    def to_parfile_config(self):
        return {
            "format": self.output_format,
            "directory": self.output_folder,
            "field": self.displayed_field,
            "simulation-field": self.simfield,
            "time-interval": self.steps_between_display,
        }


@dataclass
class WavefieldOutputConfiguration:
    output_format: str = "HDF5"
    output_folder: str = "OUTPUT_FILES/wavefield"

    def to_parfile_config(self):
        return {
            "format": self.output_format,
            "directory": self.output_folder,
        }


@dataclass
class SimulationSourcesConfiguration:
    source_file: str | None = None
    source_list: list[SourceConfiguration] | None = None

    def to_parfile_config(self):
        if (self.source_file is None) == (self.source_list is None):
            e = ValueError(
                "Only one of `source_file` and `source_list` can be specified!"
            )
            raise e

        if self.source_file is not None:
            return self.source_file

        assert (
            self.source_list is not None
        )  # to silence typechecker; already checked
        return {
            "number-of-sources": len(self.source_list),
            "sources": [
                src.to_specfem_parameters() for src in self.source_list
            ],
        }


@dataclass
class SPECFEMConfiguration:
    title: str
    description: str

    quadrature_rule: PresetQuadratureRule

    dt: float
    num_steps: int

    seismo_config: SiesmogramConfiguration
    display_config: DisplayConfiguration | None
    wavefield_config: WavefieldOutputConfiguration | None
    sources_config: SimulationSourcesConfiguration
    database_input_file: str

    nproc: int = 1
    nruns: int = 1

    def export(self, config_file: str | PathLike):
        writer = {
            "seismogram": {
                "format": "ascii",
                "directory": self.seismo_config.output_folder,
            },
        }
        if self.display_config is not None:
            writer["display"] = self.display_config.to_parfile_config()
        if self.wavefield_config is not None:
            writer["wavefield"] = self.wavefield_config.to_parfile_config()

        yaml_contents = {
            "parameters": {
                "header": {
                    "title": self.title,
                    "description": self.description,
                },
                "simulation-setup": {
                    "quadrature": {
                        "quadrature-type": str(self.quadrature_rule)
                    },
                    "solver": {
                        "time-marching": {
                            "type-of-simulation": "forward",
                            "time-scheme": {
                                "type": "Newmark",
                                "dt": f"{self.dt:.5e}",
                                "nstep": f"{self.num_steps:d}",
                            },
                        }
                    },
                    "simulation-mode": {"forward": {"writer": writer}},
                },
                "receivers": {
                    "stations": self.seismo_config.stations_input_file,
                    "angle": f"{self.seismo_config.receiver_angle:.5f}",
                    "seismogram-type": [
                        str(t) for t in self.seismo_config.seismo_types
                    ],
                    "nstep_between_samples": (
                        self.seismo_config.steps_between_samples
                    ),
                },
                "run-setup": {
                    "number-of-processors": self.nproc,
                    "number-of-runs": self.nruns,
                },
                "databases": {"mesh-database": self.database_input_file},
                "sources": self.sources_config.to_parfile_config(),
            }
        }
        with Path(config_file).open("w") as f:
            dump(yaml_contents, f, Dumper)
