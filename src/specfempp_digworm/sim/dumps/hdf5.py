import re
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

from .dump_data import DumpData, Medium, MediumData

_KEY_COORD_GROUP = "Coordinates"
_KEY_COORD_X = "X"
_KEY_COORD_Z = "Z"
_KEY_COORD_MAPPING = "mapping"
_KEY_STEP_GROUP_REGEX = re.compile(r"Step(\d+)")

_KEY_BDVALS_GROUP = "BoundaryValues"
_KEY_BDVALS_SAVE = "save_boundary_values"

_KEY_MEDIUM_TAGS = "medium_tags"


@contextmanager
def thorough_group_read(group):
    @dataclass(frozen=True)
    class GroupCapture:
        unused_keys: set[str]
        _group: h5py.Group

        def __getitem__(self, key):
            self.unused_keys.remove(key)
            return self._group[key]

        def get(
            self,
            key,
            errormsg: str = "Group {group} expects key {key},"
            " but it was not found.",
        ):
            if key not in self._group:
                e = ValueError(
                    errormsg.replace("{group}", str(self._group.name)).replace(
                        "{key}", str(key)
                    )
                )
                raise e

            return self[key]

        def get_dataset(
            self,
            key,
            missing_errormsg: str = "Group {group} expects key {key}, "
            "but it was not found.",
            wrongtype_errormsg: str = "Group {group} expects key {key} to "
            "be a dataset. Was {type} instead.",
        ):
            if key not in self._group:
                e = KeyError(
                    missing_errormsg.replace(
                        "{group}", str(self._group.name)
                    ).replace("{key}", str(key))
                )
                raise e

            item = self[key]
            if not isinstance(item, h5py.Dataset):
                e = ValueError(
                    wrongtype_errormsg.replace("{group}", str(self._group.name))
                    .replace("{key}", str(key))
                    .replace("{type}", str(type(item)))
                )
                raise e
            return item

    gc = GroupCapture(unused_keys=set(group.keys()), _group=group)

    yield gc

    if gc.unused_keys:
        e = RuntimeError(
            f'Group "{group.name}" has unhandled data: {gc.unused_keys}'
        )
        raise e


def dump_from_hdf5(file: str | PathLike) -> DumpData:
    path = Path(file)

    # we do not check file suffix.
    with h5py.File(path, "r") as f_file, thorough_group_read(f_file) as f:
        media = tuple(
            Medium(v.decode()) for v in f.get_dataset(_KEY_MEDIUM_TAGS)
        )

        ndof_per_medium, coords, mappings = _read_coords(
            f.get(
                _KEY_COORD_GROUP,
                "File does not have group {key} for coordinates",
            ),
            media,
        )

        stored_steps, field_vals = _read_steps(f, media)
        _read_boundary_values(f.get(_KEY_BDVALS_GROUP))

        return DumpData(
            media=[
                MediumData(
                    medium=medium,
                    mapping=mappings[i],
                    coords=coords[i],
                    field_data=field_vals[i],
                    num_dof=ndof_per_medium[i],
                    num_elements=mappings[i].shape[0],
                    ngllx=mappings[i].shape[2],
                    ngllz=mappings[i].shape[1],
                )
                for i, medium in enumerate(media)
            ],
            num_stored_steps=len(stored_steps),
            num_dof=sum(ndof_per_medium),
            stored_steps=stored_steps,
        )


def _read_coords(coords_group, media):
    coords = []
    mappings = []

    with thorough_group_read(coords_group) as coords_gp:
        for medium in media:
            with thorough_group_read(coords_gp.get(medium)) as subgp:
                coords.append(
                    np.stack(
                        [
                            subgp.get_dataset(_KEY_COORD_X),
                            subgp.get_dataset(_KEY_COORD_Z),
                        ],
                        axis=-1,
                    )
                )
                mappings.append(subgp.get_dataset(_KEY_COORD_MAPPING)[...])
    num_media = len(media)
    ndof_per_medium = tuple(coords[i].shape[0] for i in range(num_media))

    return ndof_per_medium, coords, mappings


def _read_steps(hdf5file_ctx, media):
    step_groups = sorted(
        (int(keymatch.group(1)), keymatch.group(0))
        for keymatch in map(
            _KEY_STEP_GROUP_REGEX.match, hdf5file_ctx.unused_keys
        )
        if keymatch is not None
    )

    num_steps = len(step_groups)
    # step indices
    stored_steps = np.empty((num_steps,), dtype=np.uint64)

    # field values at each step, for each medium
    vals_per_medium: list = [None] * len(media)

    for stepind, (step, key) in enumerate(step_groups):
        stored_steps[stepind] = step
        # each step (hdf5file_ctx[key]) should be composed of media
        with thorough_group_read(hdf5file_ctx[key]) as stepgp:
            for imedium, medium in enumerate(media):
                with thorough_group_read(
                    stepgp.get(
                        medium,
                        errormsg=f"Expected medium {medium} in step group "
                        "{group}, but it's missing.",
                    )
                ) as mediumgp:
                    # this will depend on medium.
                    if medium in {Medium.ACOUSTIC, Medium.ELASTIC_PSV}:
                        keys_per_deriv = {
                            Medium.ELASTIC_PSV: (
                                "Displacement",
                                "Velocity",
                                "Acceleration",
                            ),
                            Medium.ACOUSTIC: (
                                "Potential",
                                "PotentialDot",
                                "PotentialDotDot",
                            ),
                        }[medium]
                        disp = mediumgp.get_dataset(keys_per_deriv[0])[...]
                        if vals_per_medium[imedium] is None:
                            # should be (stepind, derivorder, dof, dim)
                            vals_per_medium[imedium] = np.empty(
                                (num_steps, 3, *disp.shape)
                            )
                        vals_per_medium[imedium][stepind, 0, ...] = disp
                        # vel
                        vals_per_medium[imedium][stepind, 1, ...] = (
                            mediumgp.get_dataset(keys_per_deriv[1])
                        )
                        # accel
                        vals_per_medium[imedium][stepind, 2, ...] = (
                            mediumgp.get_dataset(keys_per_deriv[2])
                        )
                    else:
                        e = NotImplementedError(
                            f"hdf5 reading for medium {medium:s} not "
                            "yet implemented."
                        )
                        raise e

    return stored_steps, tuple(vals_per_medium)


def _read_boundary_values(bdvals_gp):
    with thorough_group_read(bdvals_gp) as bdvals:
        save_bdvals = bdvals.get_dataset(_KEY_BDVALS_SAVE)
        assert save_bdvals.size == 1
        if save_bdvals[0] != 0:
            e = NotImplementedError(
                "boundary value reading not yet implemented."
            )
            raise e
