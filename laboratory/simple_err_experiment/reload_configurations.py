import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from yaml import Dumper, Loader, dump, load

from specfempp_digworm.sim.meshfem_parfile.gen_parfile import ReceiverSeries

a0 = 0.5
configurations = [
    *[
        vars(
            SimpleNamespace(
                harmonic=harmonic,
                a0=a0,
                receivers=[
                    ReceiverSeries(
                        nrec=4,
                        xdeb=float(x),
                        zdeb=float(a0 * np.cos(harmonic * x) + 0.3),
                        xfin=float(x),
                        zfin=float(a0 * np.cos(harmonic * x) - 0.3),
                    ).to_dict()
                    for x in [-np.pi / 2, -np.pi / 4, 0, np.pi / 3]
                ],
                source_locations=[{"x": 0, "z": 0}],
                vp_below=3400 / 3400,
                vs_below=1963 / 3400,
                vp_above=1500 / 3400,
                tmax=float(2 * np.pi),
                source_f0=2,
            )
        )
        for harmonic in [-1, 1, 2, 3, 5]
    ],
]
num_configurations = len(configurations)


def reload():
    config_folder = Path(__file__).parent / "configurations"
    if not config_folder.is_dir():
        config_folder.mkdir(parents=True)

    # iterate existing files
    for iconf in itertools.count(start=0, step=1):
        config_file = config_folder / f"config_{iconf}.yaml"

        config_current = (
            configurations[iconf] if iconf < len(configurations) else None
        )

        # get previous config
        config_prev = None
        if config_file.exists():
            with config_file.open("r") as f:
                config_prev = load(f, Loader)

        # break condition: neither exists
        if config_current is None and config_prev is None:
            break

        if config_current is None:
            # delete file if we don't have a configuration for it (size decreased)
            print(  # noqa: T201
                f"Configuration {iconf} no longer exists. Removing."
            )
            config_file.unlink()

        elif config_prev is None:
            # config does not yet exist
            print(  # noqa: T201
                f"Adding new configuration {iconf}."
            )
            with (config_folder / f"config_{iconf}.yaml").open("w") as f:
                dump(config_current, f, Dumper)
        else:
            # compare and replace only if new
            if config_current == config_prev:
                print(  # noqa: T201
                    f"Configuration {iconf} unmodified. Skipping."
                )
                continue

            print(f"Overwriting configuration {iconf}.")  # noqa: T201
            with (config_folder / f"config_{iconf}.yaml").open("w") as f:
                dump(config_current, f, Dumper)


def truesol_gridsim():
    return {
        "nx_above": 200,
        "nx_below": 200,
        "dt": 0.0001,
        "do_nonconforming": False,
    }


def get_gridsims(iconf: int):
    gridsims = []

    def add_param(abovefac, belowfac, nonconforming, cfl):
        nx_above = abovefac * 20
        nx_below = belowfac * 20

        speed_above = 1963 / 3400
        speed_below = 1

        # cfl = u * dt / dx

        cfl = 0.1
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
        for cfl in [0.1, 0.025, 0.00625]:
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

    harmonic = configurations[iconf]["harmonic"]

    if harmonic < 3:
        facs.extend(
            [
                (1, 4),
                (4, 1),
            ]
        )
    if harmonic < 2:
        facs.extend(
            [
                (1, 6),
                (6, 1),
            ]
        )
    if harmonic >= 3 or harmonic < 0:
        filt = 2.5

        if harmonic > 3:
            filt = 0.6
        facs = [
            (nx1, nx2)
            for nx1, nx2 in facs
            if (abs(np.log2(nx1 / nx2)) < filt + 1e-4)
        ]

    for abovefac, belowfac in facs:
        nx_above = abovefac * 20
        nx_below = belowfac * 20

        speed_above = 1963 / 3400
        speed_below = 1

        for cfl in [0.1, 0.025, 0.00625]:
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


if __name__ == "__main__":
    reload()
