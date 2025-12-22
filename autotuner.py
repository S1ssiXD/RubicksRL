from typing import Callable, Dict, Tuple


def line_search(evaluation_function: Callable[[float], float],
                low: float, high: float, step: float) -> Tuple[float, float]:
    """ Perfroms a simple line search to find the parameter value that maximizes the evaluation function.
    Args:
        evaluation_function (Callable[[float], float]): A function that takes a single float parameter and returns a float score.
        low (float): The lower bound of the parameter search space.
        high (float): The upper bound of the parameter search space.
        step (float): The step size for the search.

    Returns:
        Tuple[float, float]: The best parameter value and its corresponding score.
    """
    best_param_value = low
    best_score = evaluation_function(best_param_value)

    param = low + step
    while param <= high:
        score = evaluation_function(param)
        if score > best_score:
            best_score = score
            best_param_value = param
        param += step

    return best_param_value, best_score


def autotune_parameters(evaluation_function: Callable[..., float],
                        initial_values: Dict[str, float],
                        bounds: Dict[str, Tuple[float, float]],
                        step_size=0.1, max_iterations=5) -> Dict[str, float]:
    """ Autotunes parameters to maximize the evaluation function.
    Args:
        evaluation_function (Callable[..., float]): A function that takes parameters as keyword arguments and returns a float score.
        initial_values (Dict[str, float]): Initial values for the parameters to be tuned.
        bounds (Dict[str, Tuple[float, float]]): Bounds for each parameter as (min, max).
        step_size (float): Step size for parameter adjustments.
        max_iterations (int): Maximum number of iterations to perform.

    Returns:
        Dict[str, float]: The tuned parameters that maximize the evaluation function.
    """

    it = 0
    best_score = evaluation_function(**initial_values)
    while it < max_iterations:
        improved = False
        for param, (low, high) in bounds.items():
            def eval_func(value: float) -> float:
                params = initial_values.copy()
                params[param] = value
                return evaluation_function(**params)

            best_value, score = line_search(
                eval_func, low, high, step_size)
            if best_value != initial_values[param] and score > best_score:
                print(
                    f"Parameter '{param}' improved from {initial_values[param]} to {best_value} with score {score} (previous best: {best_score})")
                initial_values[param] = best_value
                best_score = score
                improved = True

        if not improved:
            break

        it += 1

    print(f"Autotuning completed in {it} iterations.")
    print(f"Tuned parameters: {initial_values}")
    return initial_values
