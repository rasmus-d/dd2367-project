import numpy as np
import math
from typing import List, Union

from src.general_state import GeneralState
from src.general_channels import *

def q2s (q:int) -> int:
    return pow(2, q)

def s2q (s:int) -> int:
    #TODO: Throw error if s is not 2-exponential
    return round(math.log2(s))

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
        on_sys_ch = OnSpecificSystem(m=q2s(self.pos), n=q2s(qubits_following), dim=(q2s(ch_qin),q2s(ch_qout)), ch=self.channel)
        return on_sys_ch.apply(state)

class MatrixSimulator():
    num_qubits : int
    state : GeneralState
    #TODO: Implement measurements and add to union
    channel_queue : List[QChannel]

    def __init__(self, num_qubits:int, initial_state:GeneralState = GeneralState()):
        self.num_qubits = num_qubits
        self.state = initial_state
        self.channel_queue = []

    def add(self, op : Union[Union[QChannel],List[Union[QChannel]]]) -> None:
        if isinstance(op,list):
            self.channel_queue += op
        else:
            self.channel_queue.append(op)

    def run(self) -> GeneralState:
        state = self.state
        for op in self.channel_queue:
            if isinstance(op,QChannel):
                state = op.apply(self.num_qubits, state)
            else:
                raise Exception("Type in operator queue not known.")
        self.state = state
        return state