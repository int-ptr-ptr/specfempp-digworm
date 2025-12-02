import re
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

labdir = Path(__file__).parent
sims_dir = labdir / "sims"
out_dir = labdir / "outputs"

simulations_per_config = {}

for simfol in sims_dir.iterdir():
    if not simfol.is_dir():
        continue
    if m := re.match(r"(\d+)_(\d+)_(\d+)_([\dx]+)_(True|False)", simfol.stem):
        iconf = int(m.group(1))
        nx_below = int(m.group(2))
        nx_above = int(m.group(3))
        is_nonconforming = bool(m.group(5))

        if iconf not in simulations_per_config:
            simulations_per_config[iconf] = []

        simulations_per_config[iconf].append(
            SimpleNamespace(
                iconf=iconf,
                nx_below=nx_below,
                nx_above=nx_above,
                is_nonconforming=is_nonconforming,
                path=simfol,
            )
        )


def plot_seismos(iconf):
    simulations = simulations_per_config[iconf]
    for sim in simulations:
        seismofol = sim.path / "OUTPUT_FILES/seismograms"
        for file in seismofol.iterdir():
            data = np.loadtxt(file)
            plt.plot(data[:, 0], data[:, 1])

    outfol = out_dir / "seismoplots"
    outfol.mkdir(parents=True, exist_ok=True)

    plt.savefig(outfol / f"{iconf}.png")

if __name__ == "__main__":
    for i in range(3):
        plot_seismos(i)
