from typing import List


class Variable(object):
    def __init__(self, init_value: float = 0.0, store_history: bool = True) -> None:
        """ Variable -> meta class for `Node`, `Error`, `Data` and `Action`

        Args:
            init_value (float, optional): [initial value]. Defaults to 0.0.
            store_history (bool, optional): [whether to store variable history]. Defaults to True.
        """
        self.store_history = store_history
        self._value = init_value

        if self.store_history:
            self._history = []

    def update(self, value: float):
        """ Update variable

        Args:
            value (float): [updated value for variable]
        """
        self._value = value
        if self.store_history:
            self._history.append(self._value)

    @property
    def value(self) -> float:
        """ Get current value of variable

        Returns:
            float: [current value]
        """
        return self._value

    @property
    def history(self) -> List[float]:
        """ Get history of values

        Raises:
            AttributeError: [`store_history` has been set to `False`]

        Returns:
            List[float]: [history of values]
        """
        if not self.store_history:
            raise AttributeError(f"`{self.__class__.__name__}` has `store_history` set to `False`")
        else:
            return self._history


class Node(Variable):
    def __init__(self, init_value: float = 0.0, store_history: bool = True) -> None:
        """ Node -> represents a node in a predictive coding network

        Args:
            init_value (float, optional): [initial value]. Defaults to 0.0.
            store_history (bool, optional): [whether to store variable history]. Defaults to True.
        """
        super().__init__(init_value=init_value, store_history=store_history)


class Error(Variable):
    def __init__(
        self, init_value: float = 0.0, variance: float = 1.0, store_history: bool = True
    ) -> None:
        """Node -> represents an error in a predictive coding network

        Args:
            init_value (float, optional): [initial value]. Defaults to 0.0.
            variance (float, optional): [variance of error node]. Defaults to 1.0.
            store_history (bool, optional): [whether to store variable history]. Defaults to True.
        """
        super().__init__(init_value=init_value, store_history=store_history)
        self._variance: float = variance

    def update_variance(self, variance: float):
        """ Update variance

        Args:
            variance (float): [updated variance]
        """
        self._variance = variance

    @property
    def variance(self):
        """ Retrieve variance

        Returns:
           float : [variance]
        """
        return self._variance


class Data(Variable):
    def __init__(self, init_value: float = 0.0, store_history: bool = True) -> None:
        """ Data -> represents data in a predictive coding network

        Args:
            init_value (float, optional): [initial value]. Defaults to 0.0.
            store_history (bool, optional): [whether to store variable history]. Defaults to True.
        """
        super().__init__(init_value=init_value, store_history=store_history)


class Action(Variable):
    def __init__(self, init_value: float = 0.0, store_history: bool = True) -> None:
        """ Action -> represents active nodes in a predictive coding network

        Args:
            init_value (float, optional): [initial value]. Defaults to 0.0.
            store_history (bool, optional): [whether to store variable history]. Defaults to True.
        """
        super().__init__(init_value=init_value, store_history=store_history)


class Function(object):
    """ Meta-class for generative functions """

    def forward(self, variable: Variable) -> float:
        raise NotImplementedError

    def backward(self, variable: Variable) -> float:
        raise NotImplementedError


class IdentityFunction(Function):
    def forward(self, variable: Variable) -> float:
        return variable.value

    def backward(self, variable: Variable) -> float:
        return 1.0


class InverseFunction(Function):
    def __init__(self, param: float = 1.0):
        """ Inverse function -> f(x) = -x

        Args:
            param (float, optional): Defaults to 1.0.
        """
        self._param = 1.0

    def forward(self, variable: Variable) -> float:
        """ Apply function to variable

        Args:
            variable (Variable): [variable to apply function to]

        Returns:
            float: [output of function]
        """
        return self._param * -variable.value

    def backward(self, variable: Variable) -> float:
        """ Get derivative of function with respect to variable

        Args:
            variable (Variable): [variable with which to take derivative]

        Returns:
            float: [derivative of function]
        """
        return -1.0 * self._param
