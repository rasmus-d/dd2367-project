from src.general_state import GeneralState

import numpy as np

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
    choi_matrix : Mat

    # Identity matrix of 2 state system as default
    def __init__[N:int,M:int](self, choi_matrix:Mat[N,M] = np.array([[1,0,0,1],
                                                                     [0,0,0,0],
                                                                     [0,0,0,0],
                                                                     [1,0,0,1]], dtype=np.complex128)) -> None:
        self.choi_matrix = choi_matrix


    ''' Implemented according to the evolution formula at
        https://docs.quantum.ibm.com/api/qiskit/qiskit.quantum_info.Choi'''
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

''' Zeroes off diagonal entries '''
class CompletelyDephasingChannel(Channel): 
    def __init__(self, dim) -> None:
        choi_matrix = np.zeros((dim*dim,dim*dim), dtype=np.complex128)

        for i in range(dim):
            row_col = dim * i + i
            choi_matrix[row_col][row_col] = 1

        super().__init__(choi_matrix = choi_matrix)