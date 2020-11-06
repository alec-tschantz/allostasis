from typing import Union, List

from allostatis.types import Node, Error, Data, Action, Function


def update_node(
    node: Node,
    data_errs: List[Error] = None,
    functions: List[Function] = None,
    prior_errs: List[Error] = None,
    dt: float = 0.01,
) -> Node:
    delta = 0.0
    if data_errs is not None:
        errors = [error.value / error.variance for error in data_errs]
        if functions is not None:
            assert len(errors) == len(functions), "`data_errs` must be same length as `functions`"
            errors = [
                error * function.backward(node) for (error, function) in zip(errors, functions)
            ]
        delta = delta + sum(errors)

    if prior_errs is not None:
        errors = [-error.value / error.variance for error in prior_errs]
        delta = delta + sum(errors)

    node.update(node.value + dt * (delta))
    return node


def update_error(err: Error, mu: Node, data: Union[Node, Data], function: Function = None) -> Error:
    prediction = mu.value if function is None else function.forward(mu)
    err.update(data.value - prediction)
    return err


def update_action(action: Action, err: Error, dt: float = 0.01) -> Action:
    value = action.value + dt * (-err.value / err.variance)
    action.update(value)
    return action


def calc_free_energy(errs: List[Error]) -> float:
    return 0.5 * sum([(err.value ** 2 / err.variance) for err in errs])

