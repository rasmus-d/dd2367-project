
from typing import Dict
from collections.abc import ItemsView, KeysView, ValuesView
from numpy import isclose

# Returns true if absolute(a) <= atol
def near_zero(amplitude : complex, atol:float = 1e-8) -> bool:
    # relative tolerance is not relevant since we compare with 0
    return isclose([amplitude], [0], atol=atol)[0]

class State():
    dict : Dict[int,complex]

    def __init__(self, initial_dict:Dict[int,complex] = {0:1}) -> None:
        self.dict = initial_dict

    def get(self, index : int) -> complex:
        return self.dict.get(index, 0)

    def set(self, index : int, amplitude : complex) -> None:
        if near_zero(amplitude):
            if index in self.dict:
                del self.dict[index]
        else:
            self.dict[index] = amplitude

    def items(self) -> ItemsView[int,complex]:
        return self.dict.items()

    def keys(self) -> KeysView[int]:
        return self.dict.keys()

    def values(self) -> ValuesView[complex]:
        return self.dict.values()
