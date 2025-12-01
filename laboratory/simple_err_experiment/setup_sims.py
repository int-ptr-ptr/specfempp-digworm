import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from specfempp_digworm.config.config import get_config
from specfempp_digworm.sim.meshfem_parfile.external_mesh_config import (
    ExternalMesherFileConfig,
)
from specfempp_digworm.sim.meshfem_parfile.gen_parfile import (
    InternalMesherConfig,
    MaterialModelAcousticElastic,
    MeshConfiguration,
    ReceiverSeries,
)
from specfempp_digworm.sim.source import (
    ForcingFunction,
    SourceType,
)
from specfempp_digworm.sim.specfem_parfile.gen_parfile import (
    PresetQuadratureRule,
    SeismogramTypes,
    SiesmogramConfiguration,
    SimulationSourcesConfiguration,
    SPECFEMConfiguration,
    WavefieldOutputConfiguration,
)
from specfempp_digworm.sim.topofile.gen_topo import TopographyConfig

labdir = Path(__file__).parent
# topofile_dir = labdir / "topofiles"

SIM_XMIN = -np.pi
SIM_XMAX = np.pi


@dataclass(frozen=True)
class SimpleErrExperimentConfiguration:
    harmonic: int
    ymin: float
    ymax: float
    a0: float
    Nx_below: int
    Nx_above: int
    vp_below: float
    vs_below: float
    vp_above: float
    source_f0: float
    dt: float
    num_steps: int
    is_nonconforming: bool
    workfol: Path

    def __post_init__(self):
        if not self.is_nonconforming:
            assert self.Nx_above == self.Nx_below, (
                "is_nonconforming = False, but Nx is not the same for "
                "above and below"
            )


class SimpleErrExperimentSimulation:
    @dataclass(frozen=True)
    class MeshConfig:
        xmin: float
        xmax: float
        ymin: float
        ymax: float

        nx_below: int
        nx_above: int
        nz_below: int
        nz_above: int

        is_nonconforming: bool

    _config: SimpleErrExperimentConfiguration

    _topofile_name: str
    _parfile_name: str
    _stations_filename: str
    _database_filename: str
    _specfem_config_filename: str

    _material_elastic: MaterialModelAcousticElastic
    _material_acoustic: MaterialModelAcousticElastic
    _receiver_sets: list[ReceiverSeries]

    _mesh_config: "SimpleErrExperimentSimulation.MeshConfig"
    _external_mesh_file_config: ExternalMesherFileConfig | None

    @property
    def config(self):
        return self._config

    @property
    def work_folder(self):
        return self._config.workfol

    @property
    def topography_file(self):
        return self.work_folder / self._topofile_name

    @property
    def meshfem_parfile(self):
        return self.work_folder / self._parfile_name

    @property
    def specfem_parfile(self):
        return self.work_folder / self._specfem_config_filename

    def __init__(self, config: SimpleErrExperimentConfiguration):
        self._config = config
        self._topofile_name = "topo.dat"
        self._parfile_name = "Par_file"
        self._stations_filename = "STATIONS"
        self._database_filename = "database.bin"
        self._specfem_config_filename = "specfem_config.yaml"
        self._material_elastic = MaterialModelAcousticElastic(
            rho=1, vp=self._config.vp_below, vs=self._config.vs_below
        )
        self._material_acoustic = MaterialModelAcousticElastic(
            rho=1, vp=self._config.vp_above, vs=0
        )
        self._receiver_sets = [
            # TODO
            ReceiverSeries(nrec=1, xdeb=0, zdeb=0, xfin=0, zfin=0),
        ]

        xlow = SIM_XMIN
        xhigh = SIM_XMAX

        self._mesh_config = SimpleErrExperimentSimulation.MeshConfig(
            xmin=xlow,
            xmax=xhigh,
            ymin=self._config.ymin,
            ymax=self._config.ymax,
            is_nonconforming=self._config.is_nonconforming,
            nx_below=self._config.Nx_below,
            nx_above=self._config.Nx_above,
            nz_below=int(
                np.round(
                    self._config.Nx_below
                    / (SIM_XMAX - SIM_XMIN)
                    * abs(self._config.ymin)
                )
            ),
            nz_above=int(
                np.round(
                    self._config.Nx_above
                    / (SIM_XMAX - SIM_XMIN)
                    * self._config.ymax
                )
            ),
        )

        self._external_mesh_file_config = (
            ExternalMesherFileConfig(
                self.work_folder / "OUTPUT_FILES/external_mesh",
                nonconforming_adjacencies_file="nc_adjacencies",
            )
            if self._mesh_config.is_nonconforming
            else None
        )

    def write_topofile(self, overwrite: bool = True):
        if self.topography_file.exists() and not overwrite:
            return
        if not self.work_folder.is_dir():
            self.work_folder.mkdir(parents=True)
        xpts = np.linspace(
            self._mesh_config.xmin,
            self._mesh_config.xmax,
            max(self._mesh_config.nx_below, self._mesh_config.nx_above) * 4,
        )
        bathy = self._config.a0 * np.cos(self._config.harmonic * xpts)

        topoconfig = TopographyConfig(
            [
                [
                    (self._mesh_config.xmin, self._mesh_config.ymin),
                    (self._mesh_config.xmax, self._mesh_config.ymin),
                ],
                list(zip(xpts, bathy, strict=True)),
                [
                    (self._mesh_config.xmin, self._mesh_config.ymax),
                    (self._mesh_config.xmax, self._mesh_config.ymax),
                ],
            ],
            [self._mesh_config.nz_below, self._mesh_config.nz_above],
        )

        topoconfig.export(self.topography_file)

    def write_meshfem_parfile(self, overwrite: bool = True):
        if self.meshfem_parfile.exists() and not overwrite:
            return
        if not self.work_folder.is_dir():
            self.work_folder.mkdir(parents=True)
        meshconfig = MeshConfiguration(
            title="Simple Error Experiment (Acoustic / Elastic) Generated Mesh",
            nproc=1,
            receivers=self._receiver_sets,
            materials=[self._material_elastic, self._material_acoustic],
            output_folder="OUTPUT_FILES",
            database_output_file=self._database_filename,
            stations_filename=self._stations_filename,
            tomography_file="tomo.xyz",
            external_mesher_files=self._external_mesh_file_config,
            internal_mesher_config=InternalMesherConfig(
                topography_file=self._topofile_name,
                xmin=self._mesh_config.xmin,
                xmax=self._mesh_config.xmax,
                nx=self._mesh_config.nx_below,
                do_stacey_absorbing=False,
                absorbing_bottom=False,
                absorbing_right=False,
                absorbing_left=False,
                absorbing_top=False,
                model_regions=[
                    InternalMesherConfig.Region(
                        xlow=1,
                        xhigh=self._mesh_config.nx_below,
                        ylow=1,
                        yhigh=self._mesh_config.nz_below,
                        material_index=1,
                    ),
                    InternalMesherConfig.Region(
                        xlow=1,
                        xhigh=self._mesh_config.nx_above,
                        ylow=self._mesh_config.nz_below + 1,
                        yhigh=self._mesh_config.nz_below
                        + self._mesh_config.nz_above,
                        material_index=2,
                    ),
                ],
            ),
        )
        meshconfig.export(self.meshfem_parfile)

    def write_specfem_parfile(self, overwrite: bool = True):
        if self.specfem_parfile.exists() and not overwrite:
            return
        if not self.work_folder.is_dir():
            self.work_folder.mkdir(parents=True)
        sf_config = SPECFEMConfiguration(
            title="Simple Error Experiment (Acoustic / Elastic) Generated Mesh",
            description="",
            quadrature_rule=PresetQuadratureRule.GLL4,
            dt=self._config.dt,
            num_steps=self._config.num_steps,
            seismo_config=SiesmogramConfiguration(
                seismo_types=[SeismogramTypes.PRESSURE],
                output_folder="OUTPUT_FILES/seismograms",
                stations_input_file=self._stations_filename,
            ),
            display_config=None,
            wavefield_config=WavefieldOutputConfiguration(
                output_format="HDF5", output_folder="OUTPUT_FILES/wavefield"
            ),
            sources_config=SimulationSourcesConfiguration(
                source_list=[
                    SourceType.FORCE(
                        x=0,
                        z=0,
                        forcing_function=ForcingFunction.RICKER,
                        factor=1.0,
                        tshift=0.0,
                        f0=self._config.source_f0,
                    )
                ]
            ),
            database_input_file=self._database_filename,
        )
        sf_config.export(self.specfem_parfile)
        if sf_config.display_config is not None:
            outfol = self.work_folder / sf_config.display_config.output_folder
            if not outfol.is_dir():
                outfol.mkdir(parents=True)
        if sf_config.wavefield_config is not None:
            outfol = self.work_folder / sf_config.wavefield_config.output_folder
            if not outfol.is_dir():
                outfol.mkdir(parents=True)
        outfol = self.work_folder / sf_config.seismo_config.output_folder
        if not outfol.is_dir():
            outfol.mkdir(parents=True)

    def generate_mesh(self):
        self.write_topofile(overwrite=False)
        self.write_meshfem_parfile(overwrite=False)

        config = get_config(self.work_folder)

        if self._config.is_nonconforming:
            scripts_dir: Any = config["specfem-external", "scripts"]
            assert self._external_mesh_file_config is not None
            if not self._external_mesh_file_config.base_folder.is_dir():
                self._external_mesh_file_config.base_folder.mkdir(parents=True)
            subprocess.run(
                [
                    "uv",
                    "run",
                    "gmshlayerbuilder",
                    str(self.topography_file),
                    str(self._external_mesh_file_config.base_folder),
                    "--top",
                    "acoustic_free_surface",
                    "--bottom",
                    "acoustic_free_surface",
                    "--left",
                    "acoustic_free_surface",
                    "--right",
                    "acoustic_free_surface",
                    "--materials",
                    "SF",
                ],
                cwd=scripts_dir,
                check=True,
            )

        specfem_bin: Any = config["specfem-external", "bin"]
        subprocess.run(
            [
                str(Path(specfem_bin) / "xmeshfem2D"),
                "-p",
                str(self._parfile_name),
            ],
            cwd=self.work_folder,
            check=True,
        )

    def run_sim(self):
        self.write_specfem_parfile(overwrite=False)

        config = get_config(self.work_folder)
        specfem_bin: Any = config["specfem-external", "bin"]
        subprocess.run(
            [
                str(Path(specfem_bin) / "specfem2d"),
                "-p",
                str(self._specfem_config_filename),
            ],
            cwd=self.work_folder,
            check=True,
        )


if __name__ == "__main__":
    test_config = SimpleErrExperimentSimulation(
        SimpleErrExperimentConfiguration(
            harmonic=1,
            ymin=-np.pi,
            ymax=np.pi,
            a0=1,
            Nx_below=20,
            Nx_above=30,
            vp_below=1,
            vs_below=0.6,
            vp_above=0.5,
            source_f0=1,
            dt=0.01,
            num_steps=100,
            is_nonconforming=True,
            workfol=labdir / "_test_sim",
        )
    )
    test_config.generate_mesh()
    test_config.run_sim()
