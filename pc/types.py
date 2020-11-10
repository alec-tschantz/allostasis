from typing import Optional, List


class Variable(object):
    def __init__(self, init_value: float = 0.0, store_history: bool = True):
        self.store_history = store_history
        self._value = init_value

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
            msg = f"{self.__class__.__name__} has store_history set to False"
            raise AttributeError(msg)
        else:
            return self._history


class Node(Variable):
    def __init__(
        self,
        is_fixed: bool = False,
        dt: float = 0.1,
        init_value: float = 0.0,
        store_history: bool = True,
    ):
        super().__init__(init_value=init_value, store_history=store_history)
        self._is_fixed = is_fixed
        self._dt = dt
        self._deltas: List[float] = []

    def reset(self):
        self._deltas = []

    def append_delta(self, delta: float):
        self._deltas.append(delta)

    def apply_update(self):
        delta = sum(self._deltas)
        self.update(self.value + self._dt * delta)

    @property
    def is_fixed(self) -> bool:
        return self._is_fixed

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def deltas(self) -> List[float]:
        return self._deltas


class Action(Node):
    def __init__(self, dt: float = 0.1, init_value: float = 0.0, store_history: bool = True):
        super().__init__(dt=dt, init_value=init_value, store_history=store_history)


class Param(Node):
    def __init__(
        self,
        is_fixed: bool = False,
        l_rate: float = 0.1,
        init_value: float = 0.0,
        store_history: bool = True,
    ):
        super().__init__(
            is_fixed=is_fixed, dt=l_rate, init_value=init_value, store_history=store_history
        )


class Error(Variable):
    def __init__(self, variance: float = 1.0, init_value: float = 0.0, store_history: bool = True):
        super().__init__(init_value=init_value, store_history=store_history)
        self._variance = variance

    def update_variance(self, variance: float):
        self._variance = variance

    @property
    def variance(self) -> float:
        return self._variance


class Data(Variable):
    def __init__(self, init_value: float = 0.0, store_history: bool = True):
        super().__init__(init_value=init_value, store_history=store_history)


class Function(object):
    def __init__(self, param: Optional[Param] = None):
        param = param if param is not None else Param(is_fixed=True, init_value=1.0)
        self._param = param

    def forward(self, variable: Variable) -> float:
        raise NotImplementedError

    def backward(self, variable: Variable) -> float:
        raise NotImplementedError

    def update_param(self, value: float):
        self._param.update(value)

    def __call__(self, variable: Variable) -> float:
        return self.forward(variable)

    @property
    def param(self) -> Param:
        return self._param


class LinearFunction(Function):
    def __init__(self, param: Optional[Param] = None):
        super().__init__(param)

    def forward(self, variable: Variable) -> float:
        return self.param.value * variable.value

    def backward(self, variable: Variable) -> float:
        return self.param.value


class InverseFunction(Function):
    def __init__(self, param: Optional[Param] = None):
        super().__init__(param)

    def forward(self, variable: Variable) -> float:
        return self.param.value * -variable.value

    def backward(self, variable: Variable) -> float:
        return -1.0 * self.param.value

