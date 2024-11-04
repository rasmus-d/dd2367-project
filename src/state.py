
from typing import Dict

class State():
    dict : Dict[int,complex]

    def __init__(self, initial_dict:Dict[int,complex] = {0:1}) -> None:
        self.dict = initial_dict

    def get(self, index : int) -> complex:
        return self.dict.get(index, 0)

    def set(self, index : int, amplitude : complex) -> None:
        if amplitude == 0 and index in self.dict:
            del self.dict[index]
        else:
            self.dict[index] = amplitude

    def items(self) -> Dict[int,complex]:
        return self.dict.items()
