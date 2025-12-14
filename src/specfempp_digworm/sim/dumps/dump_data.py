from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class Medium(StrEnum):
    ACOUSTIC = "acoustic"
    ELASTIC_PSV = "elastic_psv"


@dataclass(frozen=True)
class MediumData:
    medium: Medium
    mapping: np.ndarray
    coords: np.ndarray
    field_data: np.ndarray
    num_dof: int
    num_elements: int
    ngllx: int
    ngllz: int


@dataclass(frozen=True)
class DumpData:
    media: list[MediumData]
    num_dof: int
    num_stored_steps: int
    stored_steps: np.ndarray
