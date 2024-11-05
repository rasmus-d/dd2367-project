''' This module defines quantum operators, i.e. quantum gates:
    Their interface and the implementation of basic operators.
'''

from typing import Iterable, List, Tuple
from math import cos, pi, sin, sqrt
import copy

from src.state import State

def pair(target : int, control : List[int], state : State) -> Iterable[Tuple[int,int]]:
    '''
        Gets the operator pairs for any single qubit operation
        operating on given target qubit; on a given state.

        Parameters:
            target (int): The target qubit, zero indexed.
            control (List[int]): The control qubits, zero indexed.
            state  (State): The state.
    '''
    target_bin = 1 << target
    control_bin = 0 
    # compute a mask for when all controls are 1
    for c in control:
        control_bin |= 1 << c
    for idx,_ in state.items():
        other = idx ^ target_bin
        # if all control bits in idx are 1 and pair is not already in list
        if control_bin & idx == control_bin and (other > idx or other not in state.items()):
            if other < idx:
                yield (other, idx)
            else:
                yield (idx, other)
    return None


'''
A note on control bits:

A controlled unitary gate, `U`, in a 2 qubit system with control
qubit q_1, can be be written
```
|0><0| x I + |1><1| x U
```
If q_0 is the control we have
```
I x |0><0| + U x |1><1|
```
consequently, we should be able, for a 3-qubit system be able to
write applying U on q_0 with control on q_1 as:
```
|0><0| x I x I + |1><1| x U x I
```
algorithmically, assume we have a n qubit system, and we apply
our gate U on qubit i with qubit j as the control:
    Matrix M1;
    for k in 0..(i-1):
        I


'''



class Operator:
    target : int
    control : List[int]
    def __init__(self, target: int, control: List[int] = []) -> None:
        self.target = target
        self.control = control
        pass
    def apply(self, state: State) -> State:
        raise NotImplemented("this is an abstract class")

class X(Operator):
    def __init__(self, target: int, control: List[int] = []) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, state.get(j))
            new_state.set(j, state.get(i))
        return new_state

class Y(Operator):
    def __init__(self, target: int, control: List[int] = []) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, state.get(j) * -1j)
            new_state.set(j, state.get(i) * 1j)
        return new_state

class Z(Operator):
    def __init__(self, target: int, control: List[int] = []) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, state.get(i))
            new_state.set(j, -state.get(j))
        return new_state

class S(Operator):
    def __init__(self, target: int, control: List[int] = []) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, state.get(i))
            new_state.set(j, state.get(j) * 1j)
        return new_state

class T(Operator):
    def __init__(self, target: int, control: List[int] = []) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, state.get(i))
            new_state.set(j, state.get(j) * (cos(pi/4) + sin(pi/4)*1j))
        return new_state

class H(Operator):
    def __init__(self, target: int, control: List[int] = []) -> None:
        self.target = target
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, (1/sqrt(2)) * (state.get(i) + state.get(j)))
            new_state.set(j, (1/sqrt(2)) * (state.get(i) - state.get(j)))
        return new_state

class RX(Operator):
    theta : float
    def __init__(self, target: int, theta: float, control: List[int] = []) -> None:
        self.target = target
        self.theta = theta
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, state.get(i) * cos(self.theta/2) - state.get(j) * sin(self.theta/2) * 1j)
            new_state.set(j, state.get(j) * cos(self.theta/2) - state.get(i) * sin(self.theta/2) * 1j)
        return new_state

class RY(Operator):
    theta : float
    def __init__(self, target: int, theta: float, control: List[int] = []) -> None:
        self.target = target
        self.theta = theta
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, state.get(i) * cos(self.theta/2) - state.get(j) * sin(self.theta/2))
            new_state.set(j, state.get(j) * cos(self.theta/2) + state.get(i) * sin(self.theta/2))
        return new_state

class RZ(Operator):
    theta : float
    def __init__(self, target: int, theta :float, control: List[int] = []) -> None:
        self.target = target
        self.theta = theta
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, state.get(i) * (cos(-self.theta/2) + sin(-self.theta/2)*1j))
            new_state.set(j, state.get(j) * (cos(self.theta/2) + sin(self.theta/2)*1j))
        return new_state

class P(Operator):
    theta : float
    def __init__(self, target: int, theta :float, control: List[int] = []) -> None:
        self.target = target
        self.theta = theta
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        new_state = copy.deepcopy(state)
        for i, j in pair(self.target, self.control, state):
            new_state.set(i, state.get(i))
            new_state.set(j, state.get(j) * (cos(self.theta) + sin(self.theta)*1j))
        return new_state

class SWAP(Operator):
    def __init__(self, target: int, target2: int, control: List[int] = []) -> None:
        self.target = target
        self.target2 = target2
        super().__init__(target, control)
    def apply(self, state: State) -> State:
        # SWAP(x,y) = CNOT(x,y) * CNOT(y,x) * CNOT(x,y)
        new_state = X(self.target,[self.target2]).apply(
                        X(self.target2,[self.target]).apply(
                            X(self.target,[self.target2]).apply(state)
                        )
                    )
        return new_state

class Measurement():
    target : int
    def __init__(self,target):
        self.target = target

