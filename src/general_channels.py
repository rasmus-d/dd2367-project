from src.general_state import GeneralState

import numpy as np

from typing import Tuple

#TODO: Put this somewhere so that it can be used in general_state.py also
type Matrix[N:int,M:int,D:np.generic] = np.ndarray[tuple[N,M],np.dtype[D]]
type Mat[N:int,M:int] = Matrix[N,M,np.complex128]

def mul[N:int,K:int,M:int](a: Mat[N,K], b: Mat[K,M]) -> Mat[N,M]:
    r : Mat[N,M] = a @ b
    return r
'''
The parameter dim in this api always refers to the dimension of the
input- and/or output-density matrices and not to the dimensions 
of the internal choi matrix.
'''

class Channel:
    dim : tuple[int, int]
    choi_matrix : Mat

    # Identity matrix of 2 state system as default
    def __init__[N:int,M:int](self, dim:tuple[int,int], choi_matrix:Mat[N,M]) -> None:
        self.dim = dim
        self.choi_matrix = choi_matrix


    ''' Implemented according to the evolution formula at
        https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Choi
        It uses the choi_matrix on the whole GeneralState (m assumed to be 0)'''
    def apply(self, state:GeneralState) -> GeneralState:
        rho = state.density_matrix
        choi_dim, _ = self.choi_matrix.shape
        input_dim, _ = rho.shape
        output_dim = round(choi_dim / input_dim)
        rho_t = np.transpose(rho)
        rho_t_kron = np.kron(rho_t, np.identity(output_dim))
        mul_res = mul(self.choi_matrix, rho_t_kron)
        
        #Partial trace
        res = np.zeros((output_dim, output_dim), dtype=np.complex128)
        for i in range(input_dim):
            rows_cols = [*range(i * output_dim, (i+1) * output_dim)]
            res += mul_res[np.ix_(rows_cols,rows_cols)]

        return GeneralState(res)

''' Zeroes off diagonal entries. '''
class CompletelyDephasingChannel(Channel): 
    def __init__(self, dim:int) -> None:
        choi_matrix = np.zeros((dim*dim,dim*dim), dtype=np.complex128)

        for i in range(dim):
            row_col = dim * i + i
            choi_matrix[row_col][row_col] = 1

        super().__init__(dim = (dim, dim), choi_matrix = choi_matrix)

''' Models extreme noise. Outputs the completley mixed state. '''
class CompletelyDepolarizingChannel(Channel):
    def __init__(self) -> None:
        choi_matrix = np.array([[0.5,0,0,0],
                                [0,0.5,0,0],
                                [0,0,0.5,0],
                                [0,0,0,0.5]], dtype=np.complex128)
        super().__init__(dim = (2,2), choi_matrix = choi_matrix)

''' Set one qubit to 0. '''
class QubitResetChannel(Channel):
    def __init__(self) -> None:
        choi_matrix = np.array([[1,0,0,0],
                                [0,0,0,0],
                                [0,0,1,0],
                                [0,0,0,0]], dtype=np.complex128)
        super().__init__(dim = (2,2), choi_matrix = choi_matrix)

class OnStartSystem(Channel):
    '''
    n: number of states in the following 
       subsystem Z that is not changed
    ch_phi: choi channel that is to be applied to X
    '''
    n : int
    ch_phi : Channel
    def __init__(self, n:int, ch_phi:Channel) -> None:
        indim, outdim = ch_phi.dim
        self.dim = (indim * n, outdim * n)
        self.n = n
        self.ch_phi = ch_phi

    '''
    Based on:
    https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels#channels-transform-density-matrices-into-density-matrices
    '''
    def apply(self, state:GeneralState) -> GeneralState:
        # TODO: Test if choi_phi can have different size of input and output system
        choi_phi_input, choi_phi_output = self.ch_phi.dim
        output_dim = choi_phi_output * self.n
        res = np.zeros((output_dim, output_dim), dtype=np.complex128)
        for a in range(self.n):
            for b in range(self.n):
                rows = [self.n*i + a for i in range (choi_phi_input)]
                cols = [self.n*i + b for i in range (choi_phi_input)]

                rho_ab = state.density_matrix[np.ix_(rows,cols)]
                # TODO: If we forbid non-density matrices as a GeneralState we can pass a parameter
                # that allows this again (rho_ab is not a density matrix)
                phi_rho_ab = (self.ch_phi.apply(GeneralState(rho_ab))).density_matrix
                a_bar_b_ket = np.zeros((self.n, self.n), dtype=np.complex128)
                a_bar_b_ket[a][b] = 1
                kron = np.kron(phi_rho_ab, a_bar_b_ket)

                res += kron

        return GeneralState(res)

class OnEndSystem(Channel):
    '''
    m: number of states in the preceeding 
       subsystem Z that is not changed
    choi_phi: choi channel that is to be applied to X
    '''
    m : int
    ch_phi : Channel
    def __init__(self, m:int, ch_phi:Channel) -> None:
        indim, outdim = ch_phi.dim
        self.dim = (indim * m, outdim * m)
        self.m = m
        self.ch_phi = ch_phi

    '''
    Based on:
    https://learning.quantum.ibm.com/course/general-formulation-of-quantum-information/quantum-channels#channels-transform-density-matrices-into-density-matrices
    '''
    def apply(self, state:GeneralState) -> GeneralState:
        choi_phi_input, choi_phi_output = self.ch_phi.dim
        output_dim = self.m * choi_phi_output
        res = np.zeros((output_dim, output_dim), dtype=np.complex128)
        for a in range(self.m):
            for b in range(self.m):
                rows = [*range(a * choi_phi_input, (a+1) * choi_phi_input)]
                cols = [*range(b * choi_phi_input, (b+1) * choi_phi_input)]
                rho_ab = state.density_matrix[np.ix_(rows,cols)]
                # TODO: If we forbid non-density matrices as a GeneralState we can pass a parameter
                # that allows this again (rho_ab is not a density matrix)
                phi_rho_ab = (self.ch_phi.apply(GeneralState(rho_ab))).density_matrix
                a_bar_b_ket = np.zeros((self.m, self.m), dtype=np.complex128)
                a_bar_b_ket[a][b] = 1
                kron = np.kron(a_bar_b_ket, phi_rho_ab)

                res += kron

        return GeneralState(res)

#TODO: Maybe make this one the superclass, or maybe it's good to have the simple case there.
'''
Applies a channel ch to a specific intermediate system.
'''
class OnSpecificSystem(Channel):
    '''m: States in preceeding system'''
    m : int
    '''n: States in following system'''
    n : int
    '''ch: Channel to be applied to intermediate system'''
    ch : Channel

    def __init__(self, m:int, n:int, dim:tuple[int,int], ch:Channel) -> None:
        indim, outdim = dim
        self.dim = (indim * m * n, outdim * n * m)
        self.m = m
        self.n = n
        self.ch = ch

    def apply(self, state:GeneralState) -> GeneralState:
        on_start_q = OnStartSystem(n = self.n, ch_phi=self.ch)
        on_end_q = OnEndSystem(m = self.m, ch_phi=on_start_q)
        return on_end_q.apply(state)


