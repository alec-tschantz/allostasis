import uuid
from typing import Union, Optional, Dict, List

from allostatis.types import Node, Data, Action, Error, Function


class Edge(object):
    def __init__(
        self,
        from_var: Node,
        to_var: Union[Node, Data],
        err: Error,
        func: Optional[Function] = None,
        action: Optional[Action] = None,
    ):
        self.from_var = from_var
        self.to_var = to_var
        self.err = err
        self.func = func
        self.action = action

    def reset(self):
        self.from_var.reset()

    def update_error(self):
        mu, data, func, err = self.from_var, self.to_var, self.func, self.err
        prediction = mu.value if func is None else func(mu)
        err.update(data.value - prediction)

    def update_likelihood(self):
        if not self.from_var.is_fixed:
            delta = self.err.value / self.err.variance
            if self.func is not None:
                delta = delta * self.func.backward(self.from_var)
            self.from_var.append_delta(delta)

    def update_prior(self):
        if not isinstance(self.to_var, Data) and not self.to_var.is_fixed:
            delta = -self.err.value / self.err.variance
            self.to_var.append_delta(delta)

    def update_action(self):
        if self.action is not None:
            delta = -self.err.value / self.err.variance
            self.action.append_delta(delta)

    def update_param(self):
        if self.func is not None and not self.func.param.is_fixed:
            delta = self.err.value * self.func(self.from_var)
            self.func.param.append_delta(delta)

    def apply_updates(self):
        self.from_var.apply_update()
        if not isinstance(self.to_var, Data):
            self.to_var.apply_update()
        if self.action is not None:
            self.action.apply_update()
        if self.func is not None:
            self.func.param.apply_update()


class Model(object):
    def __init__(self):
        self.edges: Dict[str, Edge] = {}

    def add_connection(
        self,
        from_var: Node,
        to_var: Union[Node, Data],
        func: Optional[Function] = None,
        action: Optional[Action] = None,
        variance: float = 1.0,
    ) -> str:
        err = Error(variance=variance)
        edge = Edge(from_var, to_var, err, func, action)
        uuid = self.get_uuid()
        self.edges[uuid] = edge
        return uuid

    def update(self):
        for _, edge in self.edges.items():
            edge.reset()
            edge.update_error()

        for _, edge in self.edges.items():
            edge.update_likelihood()
            edge.update_prior()
            edge.update_action()
            edge.update_param()

        for _, edge in self.edges.items():
            edge.apply_updates()

    def get_free_energy(self) -> float:
        free_energies: List[float] = []
        for _, edge in self.edges.items():
            err = edge.err
            free_energies.append(err.value ** 2 / err.variance)
        return 0.5 * sum(free_energies)

    def get_error(self, uuid: str) -> Error:
        return self.edges[uuid].err

    @staticmethod
    def get_uuid():
        return str(uuid.uuid4())

