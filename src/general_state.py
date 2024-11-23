import numpy as np
from numpy import typing as npt

class GeneralState():
    density_matrix : npt.NDArray[np.complex128]

    #TODO: Check if it is a valid density matrix
    def __init__(self, initial_matrix:npt.NDArray[np.complex128] = np.array([[1,0],[0,0]]),dtype=np.complex128) -> None:
        self.density_matrix = initial_matrix
