from typing import Union, List

from allostatis.types import Node, Error, Data


def update_node(node: Node, likelihood_errors: List[Error] = None, dt: float = 0.01) -> Node:
    delta = 0.0
    if likelihood_errors is not None:
        delta = delta + sum([error.value / error.variance for error in likelihood_errors])
    value = node.value + dt * (delta)
    node.set_value(value)
    return node


def update_error(error: Error, mu: Node, data: Union[Node, Data]) -> Error:
    value = data.value - mu.value
    error.set_value(value)
    return error
