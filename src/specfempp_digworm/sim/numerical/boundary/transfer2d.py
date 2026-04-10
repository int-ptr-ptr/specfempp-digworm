import numpy as np

from specfempp_digworm.util.quadrature import Quadrature


class TransferFunction:
    intersection_quadrature_rule: Quadrature
    transfer_function_a: np.ndarray
    transfer_function_b: np.ndarray
