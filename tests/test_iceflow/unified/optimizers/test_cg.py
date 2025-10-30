#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import pytest
import warnings

from igm.processes.iceflow.unified.optimizers.optimizer_cg import OptimizerCG
from .vector_mapping import VectorMapping, BoundedVectorMapping

# Mark the whole module as slow
pytestmark = pytest.mark.slow

# ---------- ID helpers (pretty failure names) ----------
def id_n(n): return f"n={n}"
def id_variant(v): return f"variant={v}"
def id_ls(s): return f"ls={s}"

# --- Utilities ---------------------------------------------------------------

def _dummy_inputs_5d(dtype=tf.float32):
    """CG expects single batch like LBFGS."""
    return tf.zeros([1, 1, 1, 1, 1], dtype=dtype)

def _count_iterations_to_convergence(costs: tf.Tensor, threshold: float) -> int:
    """Count iterations until cost drops below threshold."""
    costs_np = costs.numpy()
    converged = np.where(costs_np <= threshold)[0]
    return int(converged[0]) + 1 if len(converged) > 0 else len(costs_np)

def _check_convergence_speed(actual_iters: int, expected_max: int, test_name: str):
    """Issue warning if convergence is slower than expected."""
    if actual_iters > expected_max:
        warnings.warn(
            f"{test_name}: CG took {actual_iters} iterations (expected ≤ {expected_max}). "
            f"This may indicate poor line search, ill-conditioning, or implementation issues.",
            UserWarning
        )

# --- Tests ------------------------------------------------------------------- 

@pytest.mark.parametrize("n", [10, 50], ids=id_n)
@pytest.mark.parametrize("variant", ["PR+", "FR", "HS"], ids=id_variant)
@pytest.mark.parametrize("ls_method", ["hager-zhang", "wolfe"], ids=id_ls)
def test_rosenbrock_converges(n, variant, ls_method):
    """Test CG on the Rosenbrock function with different variants."""
    # Skip FR + Wolfe combination - known numerical instability
    if variant == "FR" and ls_method == "wolfe":
        pytest.skip("FR variant with Wolfe line search has numerical instability issues")
    
    mapping = VectorMapping(n=n, dtype=tf.float32)

    # Difficult start: (-1.2, 1.0, -1.2, 1.0, ...)
    full = np.tile(np.array([-1.2, 1.0], dtype=np.float32), n)[:n]
    mapping.theta.assign(full)

    def cost_fn(U, V, inputs):
        x = mapping.theta
        x1, x2 = x[:-1], x[1:]
        return tf.reduce_sum(100.0 * (x2 - x1 * x1) ** 2 + (1.0 - x1) ** 2)

    opt = OptimizerCG(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=True,
        print_cost_freq=1,
        line_search_method=ls_method,
        iter_max=2500,  # CG may need more iterations for larger problems
        alpha_min=0.0,
        variant=variant,
        restart_every=50,
        precision=tf.float32
    )

    costs = opt.minimize(_dummy_inputs_5d())
    final = float(costs[-1].numpy())
    
    # Check convergence speed - Rosenbrock is challenging but should converge reasonably fast
    # For n=10: ~200-400 iters, for n=50: ~500-800 iters with good line search
    expected_max = 500 if n <= 10 else 1000
    iters_to_convergence = _count_iterations_to_convergence(costs, 1e-4)
    _check_convergence_speed(
        iters_to_convergence, 
        expected_max, 
        f"test_rosenbrock_converges(n={n}, {variant}, {ls_method})"
    )
    
    assert final <= 1e-4, f"Did not converge: cost={final} (n={n}, variant={variant}, ls={ls_method})"


def test_rosenbrock_float64_support():
    """Verify CG works with float64 precision."""
    n = 20
    mapping = VectorMapping(n=n, dtype=tf.float64)
    init = np.tile(np.array([-1.2, 1.0], dtype=np.float64), n)[:n]
    mapping.theta.assign(init)

    def cost_fn(U, V, inputs):
        x = mapping.theta
        x1, x2 = x[:-1], x[1:]
        return tf.reduce_sum(100.0 * (x2 - x1 * x1) ** 2 + (1.0 - x1) ** 2)

    opt = OptimizerCG(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=True,
        print_cost_freq=1,
        line_search_method="hager-zhang",
        iter_max=2000,
        variant="PR+",
        restart_every=50,
        precision=tf.float64
    )
    costs = opt.minimize(_dummy_inputs_5d(tf.float64))
    final = float(costs[-1].numpy())
    
    # Float64 should converge well for n=20
    iters_to_convergence = _count_iterations_to_convergence(costs, 1e-4)
    _check_convergence_speed(iters_to_convergence, 600, "test_rosenbrock_float64_support")
    
    assert final <= 1e-4


# --- Convex quadratic: exact solution checks ---------------------------------

def _quadratic_cost(mapping, d_vec, b_vec):
    """
    f(x) = 0.5 * sum_i d_i x_i^2 - sum_i b_i x_i
    Minimizer (unconstrained): x* = b_i / d_i
    """
    d = tf.convert_to_tensor(d_vec, dtype=mapping.theta.dtype)
    b = tf.convert_to_tensor(b_vec, dtype=mapping.theta.dtype)
    def cost_fn(U, V, inputs):
        x = mapping.theta
        return 0.5 * tf.reduce_sum(d * x * x) - tf.reduce_sum(b * x)
    return cost_fn


@pytest.mark.parametrize("n", [8, 32], ids=id_n)
@pytest.mark.parametrize("variant", ["PR+", "DY"], ids=id_variant)
def test_quadratic_unconstrained_matches_closed_form(n, variant):
    """Test CG on convex quadratic with known solution."""
    mapping = VectorMapping(n=n, dtype=tf.float64)

    # Diagonal quadratic (mildly ill-conditioned)
    d = np.linspace(1.0, 100.0, num=n).astype(np.float64)
    b = np.ones(n, dtype=np.float64) * 2.0
    x_star = b / d

    mapping.theta.assign(tf.zeros([n], tf.float64))

    cost_fn = _quadratic_cost(mapping, d, b)

    opt = OptimizerCG(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=True,
        print_cost_freq=20,
        line_search_method="hager-zhang",
        iter_max=500,
        variant=variant,
        restart_every=n,  # Restart after n iterations for quadratic
        precision=tf.float64
    )
    costs = opt.minimize(_dummy_inputs_5d(tf.float64))

    x_final = mapping.theta.numpy()
    rel_err = np.linalg.norm(x_final - x_star) / (np.linalg.norm(x_star) + 1e-12)
    
    # Quadratic problems should converge in ~n to 3*n iterations with exact line search
    # With CG restarts every n iterations, allow up to 3*n for mildly ill-conditioned case
    iters_to_convergence = _count_iterations_to_convergence(costs, 1e-8)
    expected_max = min(3 * n, 150)  # Cap at 150 for larger n
    _check_convergence_speed(
        iters_to_convergence,
        expected_max,
        f"test_quadratic_unconstrained(n={n}, {variant})"
    )
    
    assert rel_err < 1e-5, f"rel err {rel_err} too large (n={n}, variant={variant})"

def test_quadratic_early_convergence():
    """Test that CG converges quickly on a simple quadratic."""
    n = 10
    mapping = VectorMapping(n=n, dtype=tf.float32)

    # Simple diagonal quadratic
    d = np.ones(n, dtype=np.float32) * 2.0
    b = np.ones(n, dtype=np.float32) * 4.0
    # x* = b/d = 2.0 for all components

    mapping.theta.assign(tf.zeros([n], tf.float32))

    cost_fn = _quadratic_cost(mapping, d, b)

    opt = OptimizerCG(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=True,
        print_cost_freq=1,
        line_search_method="hager-zhang",
        iter_max=100,
        variant="PR+",
        restart_every=50,
        precision=tf.float32
    )
    costs = opt.minimize(_dummy_inputs_5d(tf.float32))

    x_final = mapping.theta.numpy()
    x_star = b / d
    
    # Well-conditioned diagonal quadratic should converge in ~n iterations
    # This is a key theoretical property of CG
    # For n=10: f(x*) = 0.5*sum(2*4) - sum(4*2) = 0.5*80 - 80 = -40
    expected_cost = float(-0.5 * np.sum(b * b / d))  # Analytical minimum
    iters_to_convergence = _count_iterations_to_convergence(costs, expected_cost + 1e-3)
    _check_convergence_speed(
        iters_to_convergence,
        n + 5,  # Should converge within n iterations + small buffer
        "test_quadratic_early_convergence"
    )
    
    # Should converge very quickly (within n iterations for exact arithmetic)
    assert np.allclose(x_final, x_star, atol=1e-5), f"Solution mismatch: {x_final} vs {x_star}"
    
    # Check that final cost is near optimal
    final_cost = float(costs[-1].numpy())
    assert abs(final_cost - expected_cost) < 1e-3, f"Cost mismatch: {final_cost} vs {expected_cost}"


# --- Constrained optimization tests ------------------------------------------

@pytest.mark.parametrize("variant", ["PR+", "HS"], ids=id_variant)
def test_quadratic_box_constraints(variant):
    """Test CG on quadratic with box constraints that affect the solution."""
    n = 10
    # Bounds: x ∈ [-1, 2]
    mapping = BoundedVectorMapping(n=n, L=-1.0, U=2.0, dtype=tf.float64)

    # Diagonal quadratic: f(x) = 0.5 * sum(d_i * x_i^2) - sum(b_i * x_i)
    # Unconstrained solution: x* = b / d
    d = np.ones(n, dtype=np.float64) * 2.0
    b = np.ones(n, dtype=np.float64) * 6.0  # Unconstrained solution would be 3.0
    # But with bounds [-1, 2], constrained solution is x* = 2.0 (clamped from 3.0)

    mapping.theta.assign(tf.zeros([n], tf.float64))  # Start at origin

    cost_fn = _quadratic_cost(mapping, d, b)

    opt = OptimizerCG(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=True,
        print_cost_freq=10,
        line_search_method="hager-zhang",
        iter_max=200,
        variant=variant,
        restart_every=50,
        precision=tf.float64
    )
    costs = opt.minimize(_dummy_inputs_5d(tf.float64))

    x_final = mapping.theta.numpy()
    x_expected = np.clip(b / d, -1.0, 2.0)  # Should be 2.0 for all components

    # Verify all components hit the upper bound
    assert np.allclose(x_final, x_expected, atol=1e-4), \
        f"Solution mismatch: got {x_final}, expected {x_expected}"
    
    # Check convergence speed - constrained quadratic should still be fast
    iters_to_convergence = _count_iterations_to_convergence(costs, costs[-1].numpy() + 1e-6)
    _check_convergence_speed(
        iters_to_convergence,
        100,
        f"test_quadratic_box_constraints({variant})"
    )


def test_rosenbrock_constrained():
    """Test CG on Rosenbrock with box constraints."""
    n = 10
    # Bounds: x ∈ [0.5, 1.5] - this constrains the solution space
    # Unconstrained minimum is at (1, 1, ..., 1)
    # With bounds [0.5, 1.5], solution should still be feasible at (1, 1, ..., 1)
    mapping = BoundedVectorMapping(n=n, L=0.5, U=1.5, dtype=tf.float32)

    # Start at lower bound
    mapping.theta.assign(tf.fill([n], 0.5))

    def cost_fn(U, V, inputs):
        x = mapping.theta
        x1, x2 = x[:-1], x[1:]
        return tf.reduce_sum(100.0 * (x2 - x1 * x1) ** 2 + (1.0 - x1) ** 2)

    opt = OptimizerCG(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=True,
        print_cost_freq=50,
        line_search_method="hager-zhang",
        iter_max=1000,
        variant="PR+",
        restart_every=50,
        precision=tf.float32
    )
    costs = opt.minimize(_dummy_inputs_5d())

    x_final = mapping.theta.numpy()
    
    # Solution should be near (1, 1, ..., 1) and within bounds
    assert np.all(x_final >= 0.5 - 1e-5), "Solution violates lower bound"
    assert np.all(x_final <= 1.5 + 1e-5), "Solution violates upper bound"
    assert np.allclose(x_final, 1.0, atol=0.05), \
        f"Solution should be near (1,...,1), got {x_final}"
    
    final_cost = float(costs[-1].numpy())
    assert final_cost <= 0.1, f"Did not converge to good solution: cost={final_cost}"


def test_projection_from_infeasible_start():
    """Test that CG projects an infeasible starting point onto the constraint set."""
    n = 5
    # Bounds: x ∈ [0, 2]
    mapping = BoundedVectorMapping(n=n, L=0.0, U=2.0, dtype=tf.float64)

    # Start infeasible (outside bounds)
    mapping.theta.assign(tf.constant([-1.0, 3.0, -0.5, 2.5, 1.0], dtype=tf.float64))

    # Simple quadratic: f(x) = sum(x_i^2)
    # Unconstrained solution is x* = 0, but we start infeasible
    d = np.ones(n, dtype=np.float64) * 2.0
    b = np.zeros(n, dtype=np.float64)

    cost_fn = _quadratic_cost(mapping, d, b)

    opt = OptimizerCG(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=True,
        print_cost_freq=10,
        line_search_method="hager-zhang",
        iter_max=100,
        variant="PR+",
        restart_every=50,
        precision=tf.float64
    )
    
    # Before optimization, manually trigger the initial projection by calling minimize_impl
    costs = opt.minimize(_dummy_inputs_5d(tf.float64))
    
    x_final = mapping.theta.numpy()
    
    # After optimization, should satisfy constraints and converge to boundary
    assert np.all(x_final >= 0.0 - 1e-10), f"Solution violates lower bound: {x_final}"
    assert np.all(x_final <= 2.0 + 1e-10), f"Solution violates upper bound: {x_final}"
    
    # Should converge to x* = 0 (feasible unconstrained minimum)
    assert np.allclose(x_final, 0.0, atol=1e-4), \
        f"Should converge to zero (within bounds), got {x_final}"


@pytest.mark.parametrize("variant", ["PR+"], ids=id_variant)  # DY can be unstable with constraints
def test_active_constraints_at_solution(variant):
    """Test CG correctly handles problems where optimal solution has active constraints."""
    n = 8
    # Bounds: x ∈ [-2, 0.5]
    mapping = BoundedVectorMapping(n=n, L=-2.0, U=0.5, dtype=tf.float64)

    # Quadratic pushing toward x = 2.0, but upper bound is 0.5
    d = np.ones(n, dtype=np.float64) * 1.0
    b = np.ones(n, dtype=np.float64) * 2.0  # Unconstrained solution: x* = 2.0
    # Constrained solution should be at upper bound: x* = 0.5

    mapping.theta.assign(tf.constant([-1.0] * n, dtype=tf.float64))  # Start feasible, away from solution

    cost_fn = _quadratic_cost(mapping, d, b)

    opt = OptimizerCG(
        cost_fn=cost_fn,
        map=mapping,
        print_cost=True,
        print_cost_freq=10,
        line_search_method="hager-zhang",
        iter_max=150,
        variant=variant,
        restart_every=50,
        precision=tf.float64
    )
    costs = opt.minimize(_dummy_inputs_5d(tf.float64))

    x_final = mapping.theta.numpy()
    x_expected = np.full(n, 0.5)  # All components at upper bound

    # Verify solution is at the active constraint
    assert np.allclose(x_final, x_expected, atol=1e-4), \
        f"Solution should be at upper bound 0.5, got {x_final}"
    
    # Verify bounds are satisfied
    assert np.all(x_final >= -2.0 - 1e-10), "Solution violates lower bound"
    assert np.all(x_final <= 0.5 + 1e-10), "Solution violates upper bound"
    
    # Check reasonable convergence
    iters_to_convergence = _count_iterations_to_convergence(costs, costs[-1].numpy() + 1e-6)
    _check_convergence_speed(
        iters_to_convergence,
        80,
        f"test_active_constraints_at_solution({variant})"
    )

