
from typing import Dict
from collections.abc import ItemsView, KeysView, ValuesView

class State():
    dict : Dict[int,complex]

    def __init__(self, initial_dict:Dict[int,complex] = {0:1}) -> None:
        self.dict = initial_dict

    def get(self, index : int) -> complex:
        return self.dict.get(index, 0)

    def set(self, index : int, amplitude : complex) -> None:
        if amplitude == 0 and index in self.dict:
            del self.dict[index]
        elif amplitude != 0:
            self.dict[index] = amplitude

    def items(self) -> ItemsView[int,complex]:
        return self.dict.items()

    def keys(self) -> KeysView[int]:
        return self.dict.keys()

    def values(self) -> ValuesView[complex]:
        return self.dict.values()
