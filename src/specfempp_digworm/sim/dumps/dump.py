from os import PathLike
from pathlib import Path

from ...util.GLL import GLL
from .dump_data import DumpData
from .hdf5 import dump_from_hdf5


class Dump:
    raw: DumpData
    gllx: GLL
    gllz: GLL

    def __init__(self, file: str | PathLike):
        path = Path(file)
        if path.suffix in {".h5", ".hdf5"}:
            self.raw = dump_from_hdf5(file)
        else:
            e = ValueError(f"Unsupported file type {path.suffix}")
            raise e

        ngllx = self.raw.media[0].ngllx
        ngllz = self.raw.media[0].ngllz

        self.gllx = GLL(ngllx - 1)
        self.gllz = GLL(ngllz - 1)
