from typing import Union, List

from allostatis.types import Node, Error, Data, Action, Function


def update_node(
    node: Node,
    data_errs: List[Error] = None,
    functions: List[Function] = None,
    prior_errs: List[Error] = None,
    dt: float = 0.01,
) -> Node:
    """ Update node with respect to errors

    Args:
        node (Node): [node to be update]
        data_errs (List[Error], optional): [list of data errors]. Defaults to None.
        functions (List[Function], optional): [list of generative functions]. Defaults to None.
        prior_errs (List[Error], optional): [list of prior errors]. Defaults to None.
        dt (float, optional): [delta time]. Defaults to 0.01.

    Returns:
        Node: [updated node]
    """

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
    """ Update error

    Args:
        err (Error): [error to be updated]
        mu (Node): [mean of variable associated with error]
        data (Union[Node, Data]): [data associated with error]
        function (Function, optional): [generative function]. Defaults to None.

    Returns:
        Error: [updated error]
    """
    prediction = mu.value if function is None else function.forward(mu)
    err.update(data.value - prediction)
    return err


def update_action(action: Action, err: Error, dt: float = 0.01) -> Action:
    """ Update action node

    Args:
        action (Action): [action to be updated]
        err (Error): [error]
        dt (float, optional): [delta time]. Defaults to 0.01.

    Returns:
        Action: [updated action]
    """
    value = action.value + dt * (-err.value / err.variance)
    action.update(value)
    return action


def calc_free_energy(errs: List[Error]) -> float:
    """ Calculate variational free energy

    Args:
        errs (List[Error]): [errors]

    Returns:
        float: [free energy]
    """
    return 0.5 * sum([(err.value ** 2 / err.variance) for err in errs])

