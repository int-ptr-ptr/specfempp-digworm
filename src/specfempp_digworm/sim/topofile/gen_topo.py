from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass
class TopographyConfig:
    points_per_layerbd: list[list[tuple[float, float]]]
    nz_per_layer: list[int]

    @property
    def num_layers(self):
        assert len(self.nz_per_layer) == len(self.points_per_layerbd) - 1
        return len(self.nz_per_layer)

    def export(self, topofile: str | PathLike):
        with Path(topofile).open("w") as f:
            # num interfaces
            f.write(f"{self.num_layers + 1}\n")
            for layerbd in self.points_per_layerbd:
                # for each interface: print number of points, followed by points
                f.write(f"{len(layerbd)}\n")
                for pt in layerbd:
                    f.write(f"{pt[0]} {pt[1]}\n")

            # number of cells between each interface
            for layer in self.nz_per_layer:
                f.write(f"{layer}\n")
