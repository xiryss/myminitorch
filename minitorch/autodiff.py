from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    tmp = list(vals)
    tmp[arg] += epsilon
    big = f(*tmp)
    tmp[arg] -= epsilon
    small = f(*tmp)
    return (big - small) / epsilon
    raise NotImplementedError('Need to implement for Task 1.1')


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    used = set()
    topsort = []

    def dfs(v: Variable):
        its = v.history.inputs
        used.add(v.unique_id)
        for i in its:
            if i.unique_id in used:
                continue
            dfs(i)
        topsort.append(v)
    dfs(variable)
    topsort = topsort[::-1]
    return topsort
    # TODO: Implement for Task 1.4.
    raise NotImplementedError('Need to implement for Task 1.4')


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    srt = topological_sort(variable)
    derivs = dict()
    derivs[variable.unique_id] = deriv
    for v in srt:
        if v.is_leaf():
            v.accumulate_derivative(derivs[v.unique_id])
        else:
            props = v.chain_rule(derivs[v.unique_id])
            for i, to_add in props:
                if i.unique_id in derivs.keys():
                    derivs[i.unique_id] += to_add
                else:
                    derivs[i.unique_id] = to_add
    # raise NotImplementedError('Need to implement for Task 1.4')


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
