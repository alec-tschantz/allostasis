from typing import Union, List

from allostatis.types import Node, Error, Data, Function


def update_node(
    node: Node,
    likelihood_errors: List[Error] = None,
    prior_errors: List[Error] = None,
    functions: List[Function] = None,
    delta_time: float = 0.01,
) -> Node:
    if (likelihood_errors is not None and functions is not None) and len(likelihood_errors) != len(functions):
        raise ValueError("`likelihood_errors` must be same length as `functions`")

    delta = 0.0
    if likelihood_errors is not None:
        delta = delta + sum([error.value / error.variance for error in likelihood_errors])
    if prior_errors is not None:
        errors = [-error.value / error.variance for error in prior_errors]
        if functions is not None:
            errors = [error * function.backward(node) for (error, function) in zip(errors, functions)]
        delta = delta + sum(errors)
    value = node.value + delta_time * (delta)
    node.update(value)
    return node


def update_error(error: Error, mu: Node, data: Union[Node, Data], function: Function = None) -> Error:
    prediction = mu.value if function is None else function.forward(mu)
    value = data.value - prediction
    error.update(value)
    return error
