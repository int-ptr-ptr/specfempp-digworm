import itertools
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy as sp
from yaml import Loader, load

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
    SourceConfiguration,
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
config_folder = labdir / "configurations"
# topofile_dir = labdir / "topofiles"

PT_SOURCE_FILE = labdir / "line_source.yaml"

SIM_XMIN = 0
SIM_XMAX = 20_000
SIM_YMIN = 0
SIM_YMID = 4963.982035928144
SIM_YMAX = 9_600


@dataclass(frozen=True)
class SimpleErrExperimentConfiguration:
    nx_below: int
    nx_above: int
    dt: float
    num_steps: int
    is_nonconforming: bool
    workfol: Path

    def __post_init__(self):
        if not self.is_nonconforming:
            assert self.nx_above == self.nx_below, (
                "is_nonconforming = False, but Nx is not the same for "
                "above and below"
            )


class SimpleErrExperimentSimulation:
    @dataclass(frozen=True)
    class MeshConfig:
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

    def __init__(
        self,
        config: SimpleErrExperimentConfiguration,
        receivers: list[ReceiverSeries],
    ):
        self._config = config
        self._topofile_name = "topo.dat"
        self._parfile_name = "Par_file"
        self._stations_filename = "STATIONS"
        self._database_filename = "database.bin"
        self._specfem_config_filename = "specfem_config.yaml"
        self._material_elastic = MaterialModelAcousticElastic(
            rho=2500, vp=3400, vs=1963
        )
        self._material_acoustic = MaterialModelAcousticElastic(
            rho=1020, vp=1500, vs=0
        )
        self._receiver_sets = receivers

        self._mesh_config = SimpleErrExperimentSimulation.MeshConfig(
            is_nonconforming=self._config.is_nonconforming,
            nx_below=self._config.nx_below,
            nx_above=self._config.nx_above,
            nz_below=int(
                np.round(
                    self._config.nx_below
                    * (SIM_YMID - SIM_YMIN)
                    / (SIM_XMAX - SIM_XMIN)
                )
            ),
            nz_above=int(
                np.round(
                    self._config.nx_above
                    * (SIM_YMAX - SIM_YMID)
                    / (SIM_XMAX - SIM_XMIN)
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

        loaded_topo = np.loadtxt(labdir / "topography_file_orig.dat")
        xmin = 0
        xmax = 20_000
        ymin = 0
        ymax = 9_600

        numsamples = max(self._config.nx_above, self._config.nx_below) * 3
        topo_spline = sp.interpolate.CubicSpline(
            x=loaded_topo[:, 0], y=loaded_topo[:, 1]
        )

        xsamp = np.linspace(xmin, xmax, numsamples)

        topoconfig = TopographyConfig(
            [
                [
                    (xmin, ymin),
                    (xmax, ymin),
                ],
                list(zip(xsamp, topo_spline(xsamp), strict=True)),
                [
                    (xmin, ymax),
                    (xmax, ymax),
                ],
            ],
            [self._mesh_config.nz_below, self._mesh_config.nz_above],
        )

        topoconfig.export(self.topography_file)

    def write_meshfem_parfile(self, overwrite: bool = True):
        xmin = 0
        xmax = 20_000
        ymin = 0
        ymax = 9_600

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
                xmin=xmin,
                xmax=xmax,
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
            wavefield_config=None,
            sources_config=SimulationSourcesConfiguration(
                source_file=str(PT_SOURCE_FILE)
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
        with (self.work_folder / "specfem_log.txt").open("w") as f:
            subprocess.run(
                [
                    str(Path(specfem_bin) / "specfem2d"),
                    "-p",
                    str(self._specfem_config_filename),
                ],
                cwd=self.work_folder,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )


def test_config_fol(
    *,
    nx_below: int,
    nx_above: int,
    dt: float,
    do_nonconforming: bool,
):
    return labdir / (
        f"sims/{nx_below}_{nx_above}_"
        f"{str(dt).replace('.', 'x')}_{do_nonconforming}"
    )


SIM_TMAX = 16.25


def test_config(
    *,
    nx_below: int,
    nx_above: int,
    dt: float,
    do_nonconforming: bool,
):
    num_steps = int(np.ceil(float(SIM_TMAX) / dt))

    return SimpleErrExperimentSimulation(
        SimpleErrExperimentConfiguration(
            nx_below=nx_below,
            nx_above=nx_above,
            dt=dt,
            num_steps=num_steps,
            is_nonconforming=do_nonconforming,
            workfol=test_config_fol(
                nx_below=nx_below,
                nx_above=nx_above,
                dt=dt,
                do_nonconforming=do_nonconforming,
            ),
        ),
        receivers=[
            ReceiverSeries(
                nrec=5, xdeb=11_000, zdeb=4000, xfin=11_000, zfin=6000
            ),
            ReceiverSeries(
                nrec=5, xdeb=9_000, zdeb=4000, xfin=9_000, zfin=6000
            ),
            ReceiverSeries(
                nrec=2, xdeb=10_000, zdeb=5.480e03, xfin=10_000, zfin=8.082e03
            ),
        ],
    )


def truesol_gridsim():
    nx_above = 1500
    nx_below = 1500
    # nx_above = 400
    # nx_below = 400

    speed_above = 1963 / 3400
    speed_below = 1

    # cfl = u * dt / dx

    cfl = 0.00625
    dt = cfl * np.pi * 2 / max(nx_above * speed_above, nx_below * speed_below)
    return {
        "nx_above": nx_above,
        "nx_below": nx_below,
        "dt": dt,
        "do_nonconforming": False,
    }


def get_gridsims():
    gridsims = []

    def add_param(abovefac, belowfac, nonconforming, cfl):

        for base_mult in [100, 40]:
            nx_above = abovefac * base_mult
            nx_below = belowfac * base_mult

            speed_above = 1963 / 3400
            speed_below = 1

            # cfl = u * dt / dx

            cfl = 0.00625
            dt = (
                cfl
                * np.pi
                * 2
                / max(nx_above * speed_above, nx_below * speed_below)
            )
            gridsims.append(
                {
                    "nx_above": nx_above,
                    "nx_below": nx_below,
                    "do_nonconforming": nonconforming,
                    "dt": dt,
                }
            )

    for fac in [1, 2, 4, 8]:
        for cfl in [0.025, 0.00625]:
            add_param(fac, fac, False, cfl)

    facs = [
        # lean correct
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 3),
        (7, 3),
        (7, 5),
        (8, 5),
        (9, 5),
        # lean opposite
        (3, 4),
        (3, 5),
        (3, 7),
        (5, 9),
        # roughly equal
        (1, 1),
        (2, 2),
        (6, 5),
        (9, 8),
        # extremes
        # (2, 8),
        # (8, 2),
    ]

    facs.extend(
        [
            (1, 4),
            (4, 1),
        ]
    )
    filt = 1.5
    facs = [
        (nx1, nx2)
        for nx1, nx2 in facs
        if (abs(np.log2(nx1 / nx2)) < filt + 1e-4)
    ]

    for base_mult in [100, 40]:

        for abovefac, belowfac in facs:
            nx_above = abovefac * base_mult
            nx_below = belowfac * base_mult

            speed_above = 1963 / 3400
            speed_below = 1

            for cfl in [0.025, 0.00625]:
                # cfl = u * dt / dx
                dt = (
                    cfl
                    * np.pi
                    * 2
                    / max(nx_above * speed_above, nx_below * speed_below)
                )
                gridsims.append(
                    {
                        "nx_above": nx_above,
                        "nx_below": nx_below,
                        "do_nonconforming": True,
                        "dt": dt,
                    }
                )
    
    return gridsims


# ==============================================================================
# manual run
# ==============================================================================
def run_sims():
    import sys
    import time
    from multiprocessing import Process

    queued_meshgen_tasks = []
    meshgen_tasks = []
    queued_run_tasks = []
    run_tasks = []

    max_meshgen_tasks = 6
    max_run_tasks = 1

    for sim_params in itertools.chain(
        get_gridsims(),
        [truesol_gridsim()],
    ):
        conf = test_config(**sim_params)
        if not (conf.work_folder / conf._database_filename).exists():
            print(  # noqa: T201
                f"Queued mesh generation for {sim_params}"
            )
            queued_meshgen_tasks.append(
                (sim_params, conf, Process(target=conf.generate_mesh))
            )
        elif not (conf.work_folder / "specfem_log.txt").exists():
            queued_run_tasks.append(
                (sim_params, conf, Process(target=conf.run_sim))
            )

    while (
        queued_meshgen_tasks or meshgen_tasks or queued_run_tasks or run_tasks
    ):
        changes = []

        # read completed meshgen tasks
        completed_inds = []
        for itask, (sim_params, conf, proc) in enumerate(meshgen_tasks):
            if proc.exitcode is None:
                # not yet complete
                continue

            if proc.exitcode != 0:
                exitcode = proc.exitcode
                print(f"!!! Error with mesher {conf.work_folder}. Exiting!")  # noqa: T201
                for p in meshgen_tasks:
                    p[-1].terminate()
                for p in run_tasks:
                    p[-1].terminate()
                for p in meshgen_tasks:
                    p[-1].join()
                    p[-1].close()
                for p in run_tasks:
                    p[-1].join()
                    p[-1].close()
                sys.exit(exitcode)

            completed_inds.append(itask)
            proc.join()
            proc.close()
            if not (conf.work_folder / "specfem_log.txt").exists():
                changes.append(
                    f"Queued sim run for {sim_params}\n {conf.work_folder}"
                )
                queued_run_tasks.append(
                    (sim_params, conf, Process(target=conf.run_sim))
                )
        for ind in reversed(completed_inds):
            del meshgen_tasks[ind]

        # fill meshrun task queue
        while len(meshgen_tasks) < max_meshgen_tasks and queued_meshgen_tasks:
            p = queued_meshgen_tasks.pop()
            p[-1].start()
            changes.append(f"Started mesh generation for {p[0]}")

            meshgen_tasks.append(p)

        # read completed run tasks
        completed_inds = []
        for itask, (sim_params, conf, proc) in enumerate(run_tasks):
            if proc.exitcode is None:
                # not yet complete
                continue

            if proc.exitcode != 0:
                print(f"!!! Error with runner {conf.work_folder}.")  # noqa: T201
            completed_inds.append(itask)
            proc.join()
            proc.close()
            changes.append(f"Completed simulation for {sim_params}")
        for ind in reversed(completed_inds):
            del run_tasks[ind]

        # fill run task queue
        while len(run_tasks) < max_run_tasks and queued_run_tasks:
            p = queued_run_tasks.pop()
            p[-1].start()
            changes.append(f"Started sim run for {p[0]}")

            run_tasks.append(p)

        if changes:
            updatestr = (
                "==============================\n"
                "Update:\n"
                "==============================\n"
            )

            for change in changes:
                updatestr += "  - " + change + "\n"

            updatestr += (
                "==============================\n"
                "meshes being generated:\n"
                + "".join(
                    "- " + str(p[1].work_folder) + "\n" for p in meshgen_tasks
                )
                + f"with {len(queued_meshgen_tasks)} queued.\n"
                + "==============================\nsims being run:\n"
                + "".join(
                    "- " + str(p[1].work_folder) + "\n" for p in run_tasks
                )
                + f"with {len(queued_run_tasks)} queued.\n"
                + "==============================\n"
            )
            print(updatestr)  # noqa: T201

        time.sleep(1)


if __name__ == "__main__":
    run_sims()
