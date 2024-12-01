import numpy as np
from numpy import typing as npt

class GeneralState():
    density_matrix : npt.NDArray[np.complex128]

    #TODO: Check if it is a valid density matrix
    def __init__(self, qdim = 1, initial_matrix:npt.NDArray[np.complex128] = None) -> None:
        if initial_matrix is None:
            density_matrix = np.zeros((2**qdim,2**qdim), dtype=np.complex128)
            density_matrix[0][0] = 1
            self.density_matrix = density_matrix
        else:
            self.density_matrix = initial_matrix