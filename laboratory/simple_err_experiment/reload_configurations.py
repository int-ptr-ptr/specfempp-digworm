import itertools
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from yaml import Dumper, Loader, dump, load

from specfempp_digworm.sim.meshfem_parfile.gen_parfile import ReceiverSeries


def reload():
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
            for harmonic in [1, 2, 3]
        ],
    ]

    config_folder = Path(__file__).parent / "configurations"
    if not config_folder.is_dir():
        config_folder.mkdir(parents=True)

    # iterate existing files
    for iconf in itertools.count(1):
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

if __name__ == "__main__":
    reload()