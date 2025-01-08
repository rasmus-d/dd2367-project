from src.general_state import GeneralState

import cmath
import numpy as np

from typing import Tuple

#TODO: Put this somewhere so that it can be used in general_state.py and general_simulator.py also
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

        return GeneralState(initial_matrix=res)

class IdentityChannel(Channel):
    def __init__(self, dim:int) -> None:
        choi_matrix = np.zeros((dim*dim, dim*dim), dtype=np.complex128)
        for a in range(dim):
            for b in range(dim):
                choi_matrix[a*dim + a][b*dim + b] = 1
        super().__init__((dim, dim), choi_matrix)

'''
Zeroes off diagonal entries.
This is equivalent to a standard basis measurement of the system.
'''
class CompletelyDephasingChannel(Channel):
    def __init__(self, dim:int) -> None:
        choi_matrix = np.zeros((dim*dim,dim*dim), dtype=np.complex128)

        for i in range(dim):
            row_col = dim * i + i
            choi_matrix[row_col][row_col] = 1

        super().__init__(dim = (dim, dim), choi_matrix = choi_matrix)

''' Models noise. epsilon=1 outputs the completley mixed state.'''
class DepolarizingChannel(Channel):
    def __init__(self, epsilon:float, qdim = 1) -> None:
        states = 2**qdim
        dephase_matrix = np.zeros((states*states, states*states), dtype=np.complex128)
        for a in range (states*states):
            dephase_matrix[a][a] = 1/states

        identity_matrix = IdentityChannel(states).choi_matrix

        matrix = dephase_matrix * epsilon + identity_matrix * (1-epsilon)

        super().__init__(dim = (states,states), choi_matrix = matrix)

''' Set one qubit to 0. '''
class QubitResetChannel(Channel):
    def __init__(self) -> None:
        choi_matrix = np.array([[1,0,0,0],
                                [0,0,0,0],
                                [0,0,1,0],
                                [0,0,0,0]], dtype=np.complex128)
        super().__init__(dim = (2,2), choi_matrix = choi_matrix)

class OnLeftSystem(Channel):
    '''
    n: number of states in the right 
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
                phi_rho_ab = (self.ch_phi.apply(GeneralState(initial_matrix=rho_ab))).density_matrix
                a_bar_b_ket = np.zeros((self.n, self.n), dtype=np.complex128)
                a_bar_b_ket[a][b] = 1
                kron = np.kron(phi_rho_ab, a_bar_b_ket)

                res += kron

        return GeneralState(initial_matrix=res)

class OnRightSystem(Channel):
    '''
    m: number of states in the left 
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
                phi_rho_ab = (self.ch_phi.apply(GeneralState(initial_matrix=rho_ab))).density_matrix
                a_bar_b_ket = np.zeros((self.m, self.m), dtype=np.complex128)
                a_bar_b_ket[a][b] = 1
                kron = np.kron(a_bar_b_ket, phi_rho_ab)

                res += kron

        return GeneralState(initial_matrix=res)

#TODO: Maybe make this one the superclass, or maybe it's good to have the simple case there.
'''
Applies a channel ch to a specific intermediate system.
'''
class OnSpecificSystem(Channel):
    '''n: States in right system'''
    n : int

    '''m: States in left system'''
    m : int

    '''ch: Channel to be applied to intermediate system'''
    ch : Channel

    def __init__(self, n:int, m:int, dim:tuple[int,int], ch:Channel) -> None:
        indim, outdim = dim
        self.dim = (indim * m * n, outdim * n * m)
        self.n = n
        self.m = m
        self.ch = ch

    def apply(self, state:GeneralState) -> GeneralState:
        on_left_q = OnLeftSystem(n = self.n, ch_phi=self.ch)
        on_right_q = OnRightSystem(m = self.m, ch_phi=on_left_q)
        return on_right_q.apply(state)


class Unitary(Channel):
    matrix : Mat
    def __init__[N:int,M:int](self, matrix:Mat[N,M]) -> None:
        self.matrix = matrix
        super().__init__(dim = matrix.shape, choi_matrix = None)

    def apply(self, state:GeneralState) -> GeneralState:
        mul1 = mul(state.density_matrix, np.array(np.matrix(self.matrix).H))
        return GeneralState(initial_matrix = mul(self.matrix, mul1))

class HChannel(Unitary):
    def __init__(self) -> None:
        matrix = np.array([[1/np.sqrt(2),1/np.sqrt(2)],
                           [1/np.sqrt(2),-1/np.sqrt(2)]]
                           , dtype=np.complex128)
        super().__init__(matrix = matrix)

class XChannel(Unitary):
    def __init__(self) -> None:
        matrix = np.array([[0,1],
                           [1,0]]
                           , dtype=np.complex128)
        super().__init__(matrix = matrix)

class PChannel(Unitary):
    def __init__(self, theta:float) -> None:
        matrix = np.array([[1,0],
                           [0,cmath.exp(1j*theta)]]
                           , dtype=np.complex128)
        super().__init__(matrix = matrix)

class SwapChannel(Unitary):
    def __init__(self, qdist=1) -> None:
        qubits = qdist+1
        states = pow(2, qubits)
        matrix = np.zeros((states, states), dtype=np.complex128)
        for old_state in range(states):
            old_bin = list(('{0:0'+str(qubits)+'b}').format(old_state))
            new_bin = old_bin.copy()
            new_bin[0] = old_bin[qubits-1]

            new_bin[qubits-1] = old_bin[0]
            new_state = int("".join(str(x) for x in new_bin), 2)
            matrix[new_state][old_state] = 1
        super().__init__(matrix = matrix)

class UnitaryOnSpecificSystem(Unitary):
    '''n: States in right system'''
    n : int
    '''m: States in left system'''
    m : int
    '''uni: Unitary to be applied to intermediate system'''
    uni : Unitary

    def __init__(self, n:int, m:int, uni:Unitary) -> None:
        self.n = n
        self.m = m
        self.uni = uni

    def apply(self, state:GeneralState, ret_channel:bool = False) -> GeneralState:
        composed_matrix = np.kron(np.identity(self.m), np.kron(self.uni.matrix, np.identity(self.n)))
        composed_uni = Unitary(matrix = composed_matrix)
        if ret_channel:
            return composed_uni
        else:
            return composed_uni.apply(state)