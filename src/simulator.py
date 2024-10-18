'''
This module defines the quantum simulato interface:
How to create simulators, how to add to its circuit
and how to run them.
'''

from typing import Dict, Union
from random import choices

from src.state import State
from src.operators import *

class QuantumSimulator():
    num_qubits : int
    state : State
    # this queue can later be optimized; reordered etc
    operator_queue : List[Union[Operator, Measurement]]

    def __init__(self,num_qubits,initial_state={0:1}) -> None:
        self.num_qubits = num_qubits
        self.state = initial_state
        self.operator_queue = []

    def add(self, op : Union[Operator, Measurement]) -> None:
        self.operator_queue.append(op)

    def run(self,shots:int=1) -> List[Dict[int,bool]]:
        '''
            TODO: This needs a refactoring.

            run the quantum simulator.

            makes "shots" measurements. returns a list of
            dictionaries. These dictionaries are mappings
            that tell if the measured, collapsed, qubit
            is high (True) or low (False).
        '''
        state = self.state
        meas : List[bool] = [False for _ in range(self.num_qubits)]
        for op in self.operator_queue:
            # this dynamic dispatch is a bit stupid, but should work
            if isinstance(op, Measurement):
                meas[op.target] = True
            elif isinstance(op,Operator):
                assert not meas[op.target]
                state = op.apply(state)
            else:
                raise Exception("Really bad error")

        probs = [(c**2).real for c in state.values()]
        res = choices(list(state.keys()), weights=probs, k = shots)
        r = []
        for k in range(shots):
            d = {q:bool(res[k] & (1 << q)) for q, b in enumerate(meas) if b}
            r.append(d)
        return r
