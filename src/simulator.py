'''
This module defines the quantum simulato interface:
How to create simulators, how to add to its circuit
and how to run them.
'''

from typing import Dict, Union
from random import choices

from src.state import State
from src.operators import *

import numpy as np
import math, cmath
import matplotlib
from matplotlib import pyplot as plt, patches

class QuantumSimulator():
    num_qubits : int
    state : State
    # this queue can later be optimized; reordered etc
    operator_queue : List[Union[Operator, Measurement]]

    def __init__(self, num_qubits:int ,initial_state:State = State()) -> None:
        self.num_qubits = num_qubits
        self.state = initial_state
        self.operator_queue = []

    def add(self, op : Union[Union[Operator, Measurement],List[Union[Operator, Measurement]]]) -> None:
        if isinstance(op,list):
            self.operator_queue += op
        else:
            self.operator_queue.append(op)

    def circplot(self):
        n_states = 2**self.num_qubits
        probs = {k:np.absolute(v) for k,v in self.state.items()} 
        phases = {k:np.angle(v) for k,v in self.state.items()}
        rows = int(math.ceil(n_states / 8.0))
        cols = min(n_states, 8)
        fig, axs = plt.subplots(rows, cols, squeeze=False)
        for row in range(rows):
            for col in range(cols):
                # amplitude area
                circleExt = patches.Circle((0.5, 0.5), 0.5, color='gray',alpha=0.1)
                circleInt = patches.Circle((0.5, 0.5), probs.get(8*row + col,0)/2, color='b',alpha=0.3)
                axs[row][col].add_patch(circleExt)
                axs[row][col].add_patch(circleInt)
                axs[row][col].set_aspect('equal')
                state_number = "|" + str(8*row + col) + ">"
                axs[row][col].set_title(state_number)
                xl = [0.5, 0.5 + 0.5*probs.get(8*row + col,0)*math.cos(phases.get(8*row + col,0) + np.pi/2)]
                yl = [0.5, 0.5 + 0.5*probs.get(8*row + col,0)*math.sin(phases.get(8*row + col,0) + np.pi/2)]
                axs[row][col].plot(xl,yl,'r')
                axs[row][col].axis('off')
        plt.show()

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
        self.state = state
        probs = [abs(c)**2 for c in state.values()]
        res = choices(list(state.keys()), weights=probs, k = shots)
        r = []
        for k in range(shots):
            d = {q:bool(res[k] & (1 << q)) for q, b in enumerate(meas) if b}
            r.append(d)
        return r
