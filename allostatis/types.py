from typing import List


class Variable(object):
    def __init__(self, init_value: float = None, store_history: bool = True) -> None:
        self.store_history = store_history
        self._value = 0.0 if init_value is None else init_value

        if self.store_history:
            self._history = []

    def update(self, value: float):
        self._value = value
        if self.store_history:
            self._history.append(self._value)

    @property
    def value(self) -> float:
        return self._value

    @property
    def history(self) -> List[float]:
        if not self.store_history:
            raise AttributeError(f"`{self.__class__.__name__}` has `store_history` set to `False`")
        else:
            return self._history


class Node(Variable):
    def __init__(self, init_value: float = None, store_history: bool = True) -> None:
        super().__init__(init_value=init_value, store_history=store_history)


class Error(Variable):
    def __init__(self, init_value: float = None, variance: float = 1.0, store_history: bool = True) -> None:
        super().__init__(init_value=init_value, store_history=store_history)
        self._variance: float = variance

    def update_variance(self, variance: float):
        self._variance = variance

    @property
    def variance(self):
        return self._variance


class Data(Variable):
    def __init__(self, init_value: float = None, store_history: bool = True) -> None:
        super().__init__(init_value=init_value, store_history=store_history)


class Function(object):
    def forward(self, variable: Variable) -> float:
        raise NotImplementedError

    def backward(self, variable: Variable) -> float:
        raise NotImplementedError


class LinearFunction(Function):
    def forward(self, variable: Variable) -> float:
        return variable.value

    def backward(self, variable: Variable) -> float:
        return 1.0
