import numpy as np
import math
from typing import List, Union
from numpy import typing as npt

from src.general_state import GeneralState
from src.general_channels import *

#TODO: Put this somewhere so that it can be used in general_state.py also
type Matrix[N:int,M:int,D:np.generic] = np.ndarray[tuple[N,M],np.dtype[D]]
type Mat[N:int,M:int] = Matrix[N,M,np.complex128]

def q2s (q:int) -> int:
    return pow(2, q)

def s2q (s:int) -> int:
    #TODO: Throw error if s is not 2-exponential
    return round(math.log2(s))

def mul[N:int,K:int,M:int](a: Mat[N,K], b: Mat[K,M]) -> Mat[N,M]:
    r : Mat[N,M] = a @ b
    return r

class GeneralMeasurement():

    matrices:List[Mat]

    def __init__(self, matrices:List[Mat]) -> None:
        self.matrices = matrices

    def measure(self, state:GeneralState) -> npt.NDArray:
        m, _ = state.density_matrix.shape
        probs = np.zeros((m, 1), dtype=np.float32)
        for a in range(m):
            mul_res = mul(self.matrices[a], state.density_matrix)
            probs[a] = np.trace(mul_res).real
        return probs

class StandardBasisMeasurement(GeneralMeasurement):
    def __init__(self, qdim:int) -> None:
        matrices = []
        states = q2s(qdim)
        for s in range(states):
            matrix = np.zeros((states, states), dtype=np.complex128)
            matrix[s][s] = 1
            matrices.append(matrix)
        super().__init__(matrices = matrices)

class QChannel():
    pos:int
    qdim:tuple[int,int]
    channel:Channel

    def __init__(self, pos:int, channel:Channel):
        self.pos = pos
        num_states_in, num_states_out = channel.dim
        self.qdim = (s2q(num_states_in), s2q(num_states_out))
        self.channel = channel

    def apply(self, num_qubits:int, state:GeneralState) -> GeneralState:
        ch_qin, ch_qout = self.qdim
        qubits_following = num_qubits - self.pos - ch_qin
        m = q2s(self.pos)
        n = q2s(qubits_following)
        if isinstance(self.channel, Unitary):
            on_sys_ch = UnitaryOnSpecificSystem(m = m, n = n, uni=self.channel)
        else:
            on_sys_ch = OnSpecificSystem(m = m, n = n, dim=(q2s(ch_qin),q2s(ch_qout)), ch=self.channel)
        return on_sys_ch.apply(state)

class MatrixSimulator():
    num_qubits : int
    state : GeneralState
    #TODO: Implement measurements and add to union
    operator_queue : List[QChannel]

    def __init__(self, num_qubits:int, initial_state:GeneralState = GeneralState()):
        self.num_qubits = num_qubits
        self.state = initial_state
        self.operator_queue = []

    def add(self, op : Union[Union[QChannel],List[Union[QChannel]]]) -> None:
        if isinstance(op,list):
            self.operator_queue += op
        else:
            self.operator_queue.append(op)

    def run(self) -> Union[GeneralState, np.ndarray]:
        state = self.state
        for op in self.operator_queue:
            if isinstance(op,QChannel):
                state = op.apply(self.num_qubits, state)
            elif isinstance(op, GeneralMeasurement):
                problist = op.measure(state)
                return problist
            else:
                raise Exception("Type in operator queue not known.")
        self.state = state
        return state

'''
TODO:
Measurements:
    The diagonal (Stnadard basis measurement?) alone.
    Return as probability vector or
    run shots times and return count vector.
'''