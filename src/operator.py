from typing import Dict, Iterable, List, Tuple
type State = Dict[int,complex]


def pair(target : int, state : State) -> Iterable[Tuple[int,int]]:
    '''
        Gets the operator pairs for any single qubit operation
        operating on given target qubit; on a given state.

        Parameters:
            target (int): The target qubit, zero indexed.
            state  (State): The state.
    '''
    target_idx = 1 << target
    for idx,_ in state.items():
        zero_idx = idx & ~target_idx
        one_idx = idx | target_idx
        yield (zero_idx,one_idx)
    return None


class Operator:
    def __init__(self) -> None:
        pass
    def apply(self, state: State, target: int) -> State:
        raise NotImplemented()

class X(Operator):
    def __init__(self) -> None:
        super().__init__()
    def apply(self, state: State, target: int) -> State:
        for i, j in pair(target, state):
            aux = state[i]
            state[i] = state[j]
            state[j] = aux
        raise NotImplemented()

class Y(Operator):
    def __init__(self) -> None:
        super().__init__()
    def apply(self, state: State, target: int) -> State:
        for i, j in pair(target, state):
            aux = state[i]
            state[i] = state[j] * (1j if i & (~target) else -1j)
            state[j] = aux * (1j if j & (~target) else -1j)
        raise NotImplemented()

class Z(Operator):
    def __init__(self) -> None:
        super().__init__()
    def apply(self, state: State, target: int) -> State:
        raise NotImplemented()

class H(Operator):
    def __init__(self) -> None:
        super().__init__()
    def apply(self, state: State, target: int) -> State:
        raise NotImplemented()

class R(Operator):
    theta : float
    def __init__(self, theta) -> None:
        self.theta = theta
        super().__init__()
    def apply(self, state: State, target: int) -> State:
        raise NotImplemented()


class QuantumSimulator():
    num_qubits : int
    circuit : Dict[int,List[Operator]]
    def __init__(self) -> None:
        pass

