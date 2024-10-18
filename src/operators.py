''' This module defines quantum operators, i.e. quantum gates:
    Their interface and the implementation of basic operators.
'''

from typing import Iterable, List, Tuple
from math import cos, pi, sin, sqrt

from src.state import State

def pair(target : int, state : State) -> Iterable[Tuple[int,int]]:
    '''
        Gets the operator pairs for any single qubit operation
        operating on given target qubit; on a given state.

        Parameters:
            target (int): The target qubit, zero indexed.
            state  (State): The state.
    '''

    target_bin = 1 << target

    for idx,_ in state.items():
        yield (idx, idx ^ target_bin)
    return None


class Operator:
    target : int
    control : List[int]
    def __init__(self, target: int, control: List[int]) -> None:
        self.target = target
        self.control = control
        pass
    def apply(self, _: State) -> State:
        raise NotImplemented("this is an abstract class")

class X(Operator):
    def __init__(self, target: int, control: List[int]) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = state.copy()
        for i, j in pair(self.target, state):
            new_state[i] = state.get(j,0)
            new_state[j] = state.get(i,0)
        return new_state

class Y(Operator):
    def __init__(self, target: int, control: List[int]) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = state.copy()
        for i, j in pair(self.target, state):
            new_state[i] = state.get(j,0) * -1j
            new_state[j] = state.get(i,0) * 1j
        return new_state

class Z(Operator):
    def __init__(self, target: int, control: List[int]) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = state.copy()
        for i, j in pair(self.target,state):
            new_state[i] = state.get(i,0)
            new_state[j] = -state.get(j,0)
        return new_state

class S(Operator):
    def __init__(self, target: int, control: List[int]) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = state.copy()
        for i, j in pair(self.target,state):
            new_state[i] = state.get(i,0)
            new_state[j] = state.get(j,0) * 1j
        return new_state

class T(Operator):
    def __init__(self, target: int, control: List[int]) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = state.copy()
        for i, j in pair(self.target,state):
            new_state[i] = state.get(i,0)
            new_state[j] = state.get(j,0) * (cos(pi/4) + sin(pi/4)*1j)
        return new_state

class H(Operator):
    def __init__(self, target: int, control: List[int]) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = state.copy()
        for i, j in pair(self.target, state):
            print(i,j)
            new_state[i] = (1/sqrt(2)) * (state.get(i,0) + state.get(j,0))
            new_state[j] = (1/sqrt(2)) * (state.get(i,0) - state.get(j,0))
        return new_state

class RX(Operator):
    theta : float
    def __init__(self, target: int, control: List[int], theta: float) -> None:
        self.target = target
        self.theta = theta
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = state.copy()
        for i, j in pair(self.target, state):
            new_state[i] = state.get(i,0) * cos(self.theta/2) - state.get(j,0) * sin(self.theta/2) * 1j
            new_state[j] = state.get(j,0) * cos(self.theta/2) - state.get(i,0) * sin(self.theta/2) * 1j
        return new_state

class RY(Operator):
    theta : float
    def __init__(self, target: int, control: List[int], theta: float) -> None:
        self.target = target
        self.theta = theta
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = state.copy()
        for i, j in pair(self.target, state):
            new_state[i] = state.get(i,0) * cos(self.theta/2) - state.get(j,0) * sin(self.theta/2)
            new_state[j] = state.get(j,0) * cos(self.theta/2) + state.get(i,0) * sin(self.theta/2)
        return new_state

class RZ(Operator):
    theta : float
    def __init__(self, target: int, control: List[int], theta: float) -> None:
        self.target = target
        self.theta = theta
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = state.copy()
        for i, j in pair(self.target, state):
            new_state[i] = state.get(i,0) * (cos(-self.theta/2) + sin(-self.theta)*1j)
            new_state[j] = state.get(j,0) * (cos(self.theta/2) + sin(self.theta)*1j)
        return new_state


class Measurement():
    target : int
    def __init__(self,target):
        self.target = target

