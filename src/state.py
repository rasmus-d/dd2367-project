
from typing import Dict


#type State = Dict[int,complex]

class State:
    amps : Dict[int,complex]
    dim : int
    def __init__(self,amps,dim) -> None:
        self.amps = {k:v for k,v in amps if v != 0}
        self.dim = dim

    def __matmul__(self,other):
        d1 = self.amps
        d2 = other.amps
        # maybe not very effecient
        return {i : d1[i]*d2[i] for i in d1.keys() if i in d2}

    def kron(self,other):
        d1 = self.amps
        d2 = other.amps
        d = {}
        for i in d1.keys():
            for j in d2.keys():
                d[i*self.dim+j] = d1[i] * d[j]





