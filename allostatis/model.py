from typing import Union, List

from allostatis.types import Node, Data, Error
from allostatis.functions import update_node, update_error


class Edge(object):
    def __init__(self, from_var: Node, to_var: Union[Node, Data], err: Error):
        self.from_var = from_var
        self.to_var = to_var
        self.err = err

    def reset(self):
        self.from_var.reset()

    def update_data(self):
        delta = self.err.value / self.err.variance
        self.from_var.append_delta(delta)

    def update_prior(self):
        if not isinstance(self.to_var, Data):
            delta = -self.err.value / self.err.variance
            self.to_var.append_delta(delta)

    def apply_updates(self):
        self.from_var.apply_update()
        if not isinstance(self.to_var, Data):
            self.to_var.apply_update()


class Model(object):
    def __init__(self):
        self.edges: List[Edge] = []

    def add_connection(self, from_var: Node, to_var: Union[Node, Data]):
        err = Error()
        connection = Edge(from_var, to_var, err)
        self.edges.append(connection)

    def update(self):
        self.update_errors()
        self.update_nodes()

    def update_errors(self):
        for edge in self.edges:
            edge.err = update_error(edge.err, edge.from_var, edge.to_var)

    def update_nodes(self):
        for edge in self.edges:
            edge.reset()

        for edge in self.edges:
            edge.update_data()
            edge.update_prior()

        for edge in self.edges:
            edge.apply_updates()

