from dataclasses import dataclass, field

import numpy as np

from specfempp_digworm.util.quadrature import Quadrature


@dataclass(frozen=True)
class Boundary:
    ndim: int
    quadrature_rule: Quadrature

    points: np.ndarray
    """
    An array of shape
    `stack_shape + (quadrature_rule.degree,) * (ndim - 1) + (ndim,)`
    representing the control nodes of the Boundary.
    """

    point_polys: np.ndarray = field(init=False)
    """
    An array of shape
    `stack_shape + (quadrature_rule.degree,) * (ndim - 1) + (ndim,)`
    representing the coefficients of the array.
    """

    def __post_init__(self):
        surfrange = range(self.ndim - 1)
        keys = "".join(chr(ord("a") + i) for i in surfrange)
        changestr = (
            f"...{keys}z,"
            f"{','.join(keys[i] + keys[i].upper() for i in surfrange)}"
            f"->...{keys.upper()}z"
        )
        object.__setattr__(
            self,
            "point_polys",
            np.einsum(
                changestr,
                self.points,
                *(self.quadrature_rule.L for i in surfrange),
            ),
        )
        self.points.setflags(write=False)
        self.point_polys.setflags(write=False)

    def interpolate(self, coordinate):
        """Evaluates the position field at the given local coordinates.

        `coordinate` should be an array of shape
        `stack_shape + (quadrature_rule.degree,) * (ndim - 1)`

        the return value is `(*stack_shape, ndim)`
        """
        coord_shape = np.shape(coordinate)
        coord_pows = np.reshape(coordinate, (*coord_shape, 1)) ** np.arange(
            self.quadrature_rule.degree
        )

        surfrange = range(self.ndim - 1)
        keys = "".join(chr(ord("a") + i) for i in surfrange)
        changestr = f"...{keys}z,...{keys}->...z"

        return np.einsum(changestr, self.point_polys, coord_pows)
