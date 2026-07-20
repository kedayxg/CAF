from .solver import (
    evaluate_portfolio,
    evaluate_state_vector,
    run_caf_pipeline,
    _attach_local_problem_metadata,
    _normalize_local_solution,
)

__version__ = "0.1.0"

attach_local_problem_metadata = _attach_local_problem_metadata
normalize_local_solution = _normalize_local_solution

__all__ = [
    "__version__",
    "attach_local_problem_metadata",
    "evaluate_portfolio",
    "evaluate_state_vector",
    "normalize_local_solution",
    "run_caf_pipeline",
]
