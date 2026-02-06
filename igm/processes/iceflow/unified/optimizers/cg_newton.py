# #!/usr/bin/env python3

# # Copyright (C) 2021-2025 IGM authors
# # Published under the GNU GPL (Version 3), check at the LICENSE file

# import tensorflow as tf
# from typing import Callable, Optional, Tuple

# from .optimizer import Optimizer
# from ..mappings import Mapping
# from ..halt import Halt, HaltStatus
# from .line_searches import LineSearches, ValueAndGradient


# class OptimizerCGNewton(Optimizer):
#     """
#     Matrix-free Newton–CG optimizer.

#     Outer loop: Newton
#     Inner loop: Linear CG solve of (H + damping I) p = -g
#     """

#     def __init__(
#         self,
#         cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
#         map: Mapping,
#         halt: Optional[Halt] = None,
#         print_cost: bool = True,
#         print_cost_freq: int = 1,
#         precision: str = "float32",
#         ord_grad_u: str = "l2_weighted",
#         ord_grad_theta: str = "l2_weighted",
#         line_search_method: str = "armijo",
#         alpha_min: float = 0.0,
#         iter_max: int = 100,
#         damping: float = 1e-4,
#         cg_max_iter: int = 30,
#         cg_tol: float = 1e-10,
#         **kwargs,
#     ):
#         super().__init__(
#             cost_fn,
#             map,
#             halt,
#             print_cost,
#             print_cost_freq,
#             precision,
#             ord_grad_u,
#             ord_grad_theta,
#             **kwargs,
#         )

#         self.name = "cg_newton"
#         self.line_search = LineSearches[line_search_method]()

#         self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
#         self.alpha_min = tf.Variable(alpha_min, dtype=self.precision)

#         self.damping = tf.constant(damping, dtype=self.precision)
#         self.cg_max_iter = cg_max_iter  # Keep as Python int for unrolling
#         self.cg_tol = tf.constant(cg_tol, dtype=self.precision)

#     def update_parameters(self, iter_max: int, damping: float) -> None:
#         self.iter_max.assign(iter_max)
#         self.damping = tf.cast(damping, self.precision)

#     # ============================================================
#     # Cost + gradient
#     # ============================================================

#     def _cost_and_grad(
#         self, 
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#     ):
#         with tf.GradientTape() as tape:
#             U, V = self.map.get_UV(inputs)
#             tape.watch((U, V))
#             cost = cost_fn(U, V, inputs)

#         grad_U, grad_V = tape.gradient(cost, (U, V))
#         return cost, U, V, grad_U, grad_V

#     # ============================================================
#     # Hessian–vector product using backward-over-backward
#     # ============================================================

#     def _hvp(
#         self,
#         inputs: tf.Tensor,
#         U: tf.Tensor,
#         V: tf.Tensor,
#         v_flat: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#     ) -> tf.Tensor:
#         """
#         Compute (H + damping I) v using nested GradientTapes.
#         This is backward-over-backward and avoids ForwardAccumulator issues.
#         """
        
#         nU = tf.size(U)
#         vU = tf.reshape(v_flat[:nU], tf.shape(U))
#         vV = tf.reshape(v_flat[nU:], tf.shape(V))

#         with tf.GradientTape() as outer_tape:
#             outer_tape.watch((U, V))
#             with tf.GradientTape() as inner_tape:
#                 inner_tape.watch((U, V))
#                 c = cost_fn(U, V, inputs)
#             gU, gV = inner_tape.gradient(c, (U, V))
            
#             # Compute dot product of gradient with vector
#             gv = tf.reduce_sum(gU * vU) + tf.reduce_sum(gV * vV)
        
#         # Gradient of (gradient · v) gives Hessian-vector product
#         HvU, HvV = outer_tape.gradient(gv, (U, V))

#         Hv_flat = tf.concat(
#             [tf.reshape(HvU, [-1]), tf.reshape(HvV, [-1])],
#             axis=0,
#         )

#         return Hv_flat + damping * v_flat

#     # ============================================================
#     # Linear Conjugate Gradient - UNROLLED
#     # ============================================================

#     def _cg_solve(
#         self,
#         inputs: tf.Tensor,
#         U: tf.Tensor,
#         V: tf.Tensor,
#         b: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         cg_max_iter: int,
#         cg_tol: tf.Tensor,
#     ) -> tf.Tensor:
#         """
#         Solve (H + damping I) x = b using linear CG - unrolled version.
#         """

#         x = tf.zeros_like(b)
#         r = b
#         p = r
#         rs = tf.reduce_sum(r * r)

#         # Unrolled loop using tf.range
#         for i in tf.range(cg_max_iter):
#             # Early exit if converged
#             if rs <= cg_tol:
#                 break
                
#             Ap = self._hvp(inputs, U, V, p, cost_fn, damping)
            
#             denom = tf.reduce_sum(p * Ap)
#             alpha = rs / denom
            
#             x = x + alpha * p
#             r = r - alpha * Ap
#             rs_new = tf.reduce_sum(r * r)
            
#             beta = rs_new / rs
#             p = r + beta * p
#             rs = rs_new

#         return x

#     # ============================================================
#     # Newton direction
#     # ============================================================

#     def _get_grad(
#         self, 
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         cg_max_iter: int,
#         cg_tol: tf.Tensor,
#     ):
#         """
#         Compute cost, gradient, and Newton–CG step.
#         """

#         cost, U, V, grad_U, grad_V = self._cost_and_grad(inputs, cost_fn)

#         g_flat = tf.concat(
#             [tf.reshape(grad_U, [-1]), tf.reshape(grad_V, [-1])],
#             axis=0,
#         )

#         step_flat = self._cg_solve(
#             inputs, U, V, -g_flat, cost_fn, damping, cg_max_iter, cg_tol
#         )

#         nU = tf.size(U)
#         step_U = tf.reshape(step_flat[:nU], tf.shape(U))
#         step_V = tf.reshape(step_flat[nU:], tf.shape(V))

#         return cost, (grad_U, grad_V), (step_U, step_V)

#     # ============================================================
#     # Utilities
#     # ============================================================

#     def _force_descent(
#         self, p_flat: tf.Tensor, grad_theta_flat: tf.Tensor, _: tf.Tensor
#     ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
#         dot_gp = self._dot(grad_theta_flat, p_flat)
#         return tf.cond(dot_gp >= 0.0, lambda: -grad_theta_flat, lambda: p_flat), None

#     def _apply_step(
#         self, theta_flat: tf.Tensor, alpha: tf.Tensor, p_flat: tf.Tensor
#     ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
#         return theta_flat + alpha * p_flat, None

#     @tf.function(reduce_retracing=True)
#     def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
#         dtype = self.precision
#         return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

#     # ============================================================
#     # Line search
#     # ============================================================

#     def _line_search(
#         self, 
#         theta_flat: tf.Tensor, 
#         p_flat: tf.Tensor, 
#         input: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         cg_max_iter: int,
#         cg_tol: tf.Tensor,
#     ) -> tf.Tensor:
#         def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
#             theta_backup = self.map.copy_theta(self.map.get_theta())
#             theta_alpha, _ = self._apply_step(theta_flat, alpha, p_flat)

#             self.map.set_theta(self.map.unflatten_theta(theta_alpha))
#             f, _, grad = self._get_grad(input, cost_fn, damping, cg_max_iter, cg_tol)
#             grad_flat = self.map.flatten_theta(grad)
#             df = self._dot(grad_flat, p_flat)

#             self.map.set_theta(theta_backup)
#             return ValueAndGradient(x=alpha, f=f, df=df)

#         return self.line_search.search(theta_flat, p_flat, eval_fn)

#     # ============================================================
#     # Main optimization loop
#     # ============================================================

#     @tf.function(jit_compile=False)
#     def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
#         first_batch = self.sampler(inputs)
#         if first_batch.shape[0] != 1:
#             raise NotImplementedError("Newton–CG requires a single batch.")

#         input = first_batch[0, :, :, :, :]

#         # Extract values that need to be passed as arguments
#         cost_fn = self.cost_fn
#         damping = self.damping
#         cg_max_iter = self.cg_max_iter  # Python int
#         cg_tol = self.cg_tol

#         theta_flat = self.map.flatten_theta(self.map.get_theta())

#         cost, grad_u, step_theta = self._get_grad(
#             input, cost_fn, damping, cg_max_iter, cg_tol
#         )
#         U, V = self.map.get_UV(input)
#         self._init_step_state(U, V, theta_flat)

#         halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
#         iter_last = tf.constant(-1, dtype=tf.int32)
#         costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

#         for iter in tf.range(self.iter_max):

#             cost, grad_u, step_theta = self._get_grad(
#                 input, cost_fn, damping, cg_max_iter, cg_tol
#             )

#             p_flat = self.map.flatten_theta(step_theta)
#             grad_flat = self.map.flatten_theta(grad_u)

#             p_flat, _ = self._force_descent(p_flat, grad_flat, theta_flat)

#             alpha = self._line_search(
#                 theta_flat, p_flat, input, cost_fn, damping, cg_max_iter, cg_tol
#             )
#             alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

#             theta_flat, _ = self._apply_step(theta_flat, alpha, p_flat)
#             self.map.set_theta(self.map.unflatten_theta(theta_flat))

#             costs = costs.write(iter, cost)

#             U, V = self.map.get_UV(input)
#             grad_u_norm, step_norm = self._get_grad_norm(grad_u, step_theta)
#             self._update_step_state(
#                 iter, U, V, theta_flat, cost, grad_u_norm, step_norm
#             )

#             halt_status = self._check_stopping()
#             self._update_display()

#             iter_last = iter
#             if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
#                 break

#         self._finalize_display(halt_status)
#         return costs.stack()[: iter_last + 1]

# #!/usr/bin/env python3

# # Copyright (C) 2021-2025 IGM authors
# # Published under the GNU GPL (Version 3), check at the LICENSE file

# import tensorflow as tf
# from typing import Callable, Optional, Tuple

# from .optimizer import Optimizer
# from ..mappings import Mapping
# from ..halt import Halt, HaltStatus
# from .line_searches import LineSearches, ValueAndGradient


# class OptimizerCGNewton(Optimizer):
#     """
#     Matrix-free Newton–CG optimizer.

#     Outer loop: Newton
#     Inner loop: Truncated CG solve of (H + damping I) p = -g
#     """

#     def __init__(
#         self,
#         cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
#         map: Mapping,
#         halt: Optional[Halt] = None,
#         print_cost: bool = True,
#         print_cost_freq: int = 1,
#         precision: str = "float32",
#         ord_grad_u: str = "l2_weighted",
#         ord_grad_theta: str = "l2_weighted",
#         line_search_method: str = "armijo",
#         alpha_min: float = 0.0,
#         iter_max: int = 100,
#         damping: float = 1e-3,
#         cg_max_iter: int = 30,
#         cg_tol: float = 1e-10,
#         use_truncated_cg: bool = True,
#         **kwargs,
#     ):
#         super().__init__(
#             cost_fn,
#             map,
#             halt,
#             print_cost,
#             print_cost_freq,
#             precision,
#             ord_grad_u,
#             ord_grad_theta,
#             **kwargs,
#         )

#         self.name = "cg_newton"
#         self.line_search = LineSearches[line_search_method]()

#         self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
#         self.alpha_min = tf.Variable(alpha_min, dtype=self.precision)

#         self.damping = tf.constant(damping, dtype=self.precision)
#         self.cg_max_iter = cg_max_iter  # Keep as Python int for unrolling
#         self.cg_tol = tf.constant(cg_tol, dtype=self.precision)
#         self.use_truncated_cg = use_truncated_cg

#     def update_parameters(self, iter_max: int, damping: float) -> None:
#         self.iter_max.assign(iter_max)
#         self.damping = tf.cast(damping, self.precision)

#     # ============================================================
#     # Cost + gradient
#     # ============================================================

#     def _cost_and_grad(
#         self, 
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#     ):
#         with tf.GradientTape() as tape:
#             U, V = self.map.get_UV(inputs)
#             tape.watch((U, V))
#             cost = cost_fn(U, V, inputs)

#         grad_U, grad_V = tape.gradient(cost, (U, V))
#         return cost, U, V, grad_U, grad_V

#     # ============================================================
#     # Hessian–vector product using backward-over-backward
#     # ============================================================

#     def _hvp(
#         self,
#         inputs: tf.Tensor,
#         U: tf.Tensor,
#         V: tf.Tensor,
#         v_flat: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#     ) -> tf.Tensor:
#         """
#         Compute (H + damping I) v using nested GradientTapes.
#         This is backward-over-backward and avoids ForwardAccumulator issues.
#         """
        
#         nU = tf.size(U)
#         vU = tf.reshape(v_flat[:nU], tf.shape(U))
#         vV = tf.reshape(v_flat[nU:], tf.shape(V))

#         with tf.GradientTape() as outer_tape:
#             outer_tape.watch((U, V))
#             with tf.GradientTape() as inner_tape:
#                 inner_tape.watch((U, V))
#                 c = cost_fn(U, V, inputs)
#             gU, gV = inner_tape.gradient(c, (U, V))
            
#             # Compute dot product of gradient with vector
#             gv = tf.reduce_sum(gU * vU) + tf.reduce_sum(gV * vV)
        
#         # Gradient of (gradient · v) gives Hessian-vector product
#         HvU, HvV = outer_tape.gradient(gv, (U, V))

#         Hv_flat = tf.concat(
#             [tf.reshape(HvU, [-1]), tf.reshape(HvV, [-1])],
#             axis=0,
#         )

#         return Hv_flat + damping * v_flat

#     # ============================================================
#     # Truncated Linear Conjugate Gradient (Steihaug-Tong)
#     # ============================================================

#     def _cg_solve(
#         self,
#         inputs: tf.Tensor,
#         U: tf.Tensor,
#         V: tf.Tensor,
#         b: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         cg_max_iter: int,
#         cg_tol: tf.Tensor,
#         use_truncated: bool,
#     ) -> tf.Tensor:
#         """
#         Solve (H + damping I) x = b using truncated linear CG.
        
#         Truncated CG (Steihaug-Tong) handles negative curvature by:
#         - Detecting when p^T H p <= 0 (negative curvature)
#         - Stopping early and returning the current iterate
#         - This ensures we always get a descent direction
#         """

#         x = tf.zeros_like(b)
#         r = b
#         p = r
#         rs = tf.reduce_sum(r * r)

#         # Unrolled loop using tf.range
#         for i in tf.range(cg_max_iter):
#             # Early exit if converged
#             if rs <= cg_tol:
#                 break
                
#             Ap = self._hvp(inputs, U, V, p, cost_fn, damping)
            
#             # Check for negative curvature (truncated CG)
#             pAp = tf.reduce_sum(p * Ap)
            
#             if use_truncated and pAp <= 0.0:
#                 # Negative curvature detected - return current solution
#                 # If x is still zero (first iteration), return steepest descent direction
#                 if i == 0:
#                     x = r  # Use gradient direction on first iteration
#                 break
            
#             denom = pAp
#             alpha = rs / denom
            
#             x = x + alpha * p
#             r = r - alpha * Ap
#             rs_new = tf.reduce_sum(r * r)
            
#             beta = rs_new / rs
#             p = r + beta * p
#             rs = rs_new

#         return x

#     # ============================================================
#     # Newton direction
#     # ============================================================

#     def _get_grad(
#         self, 
#         inputs: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         cg_max_iter: int,
#         cg_tol: tf.Tensor,
#         use_truncated: bool,
#     ):
#         """
#         Compute cost, gradient, and Newton–CG step.
#         """

#         cost, U, V, grad_U, grad_V = self._cost_and_grad(inputs, cost_fn)

#         g_flat = tf.concat(
#             [tf.reshape(grad_U, [-1]), tf.reshape(grad_V, [-1])],
#             axis=0,
#         )

#         step_flat = self._cg_solve(
#             inputs, U, V, -g_flat, cost_fn, damping, cg_max_iter, cg_tol, use_truncated
#         )

#         nU = tf.size(U)
#         step_U = tf.reshape(step_flat[:nU], tf.shape(U))
#         step_V = tf.reshape(step_flat[nU:], tf.shape(V))

#         return cost, (grad_U, grad_V), (step_U, step_V)

#     # ============================================================
#     # Utilities
#     # ============================================================

#     def _force_descent(
#         self, p_flat: tf.Tensor, grad_theta_flat: tf.Tensor, _: tf.Tensor
#     ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
#         dot_gp = self._dot(grad_theta_flat, p_flat)
#         return tf.cond(dot_gp >= 0.0, lambda: -grad_theta_flat, lambda: p_flat), None

#     def _apply_step(
#         self, theta_flat: tf.Tensor, alpha: tf.Tensor, p_flat: tf.Tensor
#     ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
#         return theta_flat + alpha * p_flat, None

#     @tf.function(reduce_retracing=True)
#     def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
#         dtype = self.precision
#         return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

#     # ============================================================
#     # Line search
#     # ============================================================

#     def _line_search(
#         self, 
#         theta_flat: tf.Tensor, 
#         p_flat: tf.Tensor, 
#         input: tf.Tensor,
#         cost_fn: Callable,
#         damping: tf.Tensor,
#         cg_max_iter: int,
#         cg_tol: tf.Tensor,
#         use_truncated: bool,
#     ) -> tf.Tensor:
#         def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
#             theta_backup = self.map.copy_theta(self.map.get_theta())
#             theta_alpha, _ = self._apply_step(theta_flat, alpha, p_flat)

#             self.map.set_theta(self.map.unflatten_theta(theta_alpha))
#             f, _, grad = self._get_grad(input, cost_fn, damping, cg_max_iter, cg_tol, use_truncated)
#             grad_flat = self.map.flatten_theta(grad)
#             df = self._dot(grad_flat, p_flat)

#             self.map.set_theta(theta_backup)
#             return ValueAndGradient(x=alpha, f=f, df=df)

#         return self.line_search.search(theta_flat, p_flat, eval_fn)

#     # ============================================================
#     # Main optimization loop
#     # ============================================================

#     @tf.function(jit_compile=False)
#     def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
#         first_batch = self.sampler(inputs)
#         if first_batch.shape[0] != 1:
#             raise NotImplementedError("Newton–CG requires a single batch.")

#         input = first_batch[0, :, :, :, :]

#         # Extract values that need to be passed as arguments
#         cost_fn = self.cost_fn
#         damping = self.damping
#         cg_max_iter = self.cg_max_iter  # Python int
#         cg_tol = self.cg_tol
#         use_truncated = self.use_truncated_cg

#         theta_flat = self.map.flatten_theta(self.map.get_theta())

#         cost, grad_u, step_theta = self._get_grad(
#             input, cost_fn, damping, cg_max_iter, cg_tol, use_truncated
#         )
#         U, V = self.map.get_UV(input)
#         self._init_step_state(U, V, theta_flat)

#         halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
#         iter_last = tf.constant(-1, dtype=tf.int32)
#         costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

#         for iter in tf.range(self.iter_max):

#             cost, grad_u, step_theta = self._get_grad(
#                 input, cost_fn, damping, cg_max_iter, cg_tol, use_truncated
#             )

#             p_flat = self.map.flatten_theta(step_theta)
#             grad_flat = self.map.flatten_theta(grad_u)

#             p_flat, _ = self._force_descent(p_flat, grad_flat, theta_flat)

#             alpha = self._line_search(
#                 theta_flat, p_flat, input, cost_fn, damping, cg_max_iter, cg_tol, use_truncated
#             )
#             alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

#             theta_flat, _ = self._apply_step(theta_flat, alpha, p_flat)
#             self.map.set_theta(self.map.unflatten_theta(theta_flat))

#             costs = costs.write(iter, cost)

#             U, V = self.map.get_UV(input)
#             grad_u_norm, step_norm = self._get_grad_norm(grad_u, step_theta)
#             self._update_step_state(
#                 iter, U, V, theta_flat, cost, grad_u_norm, step_norm
#             )

#             halt_status = self._check_stopping()
#             self._update_display()

#             iter_last = iter
#             if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
#                 break

#         self._finalize_display(halt_status)
#         return costs.stack()[: iter_last + 1]

#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
from typing import Callable, Optional, Tuple

from .optimizer import Optimizer
from ..mappings import Mapping
from ..halt import Halt, HaltStatus
from .line_searches import LineSearches, ValueAndGradient


class OptimizerCGNewton(Optimizer):
    """
    Matrix-free Newton–CG optimizer with diagnostics.

    Outer loop: Newton
    Inner loop: Truncated CG solve of (H + damping I) p = -g
    """

    def __init__(
        self,
        cost_fn: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
        map: Mapping,
        halt: Optional[Halt] = None,
        print_cost: bool = True,
        print_cost_freq: int = 1,
        precision: str = "float32",
        ord_grad_u: str = "l2_weighted",
        ord_grad_theta: str = "l2_weighted",
        line_search_method: str = "armijo",
        alpha_min: float = 0.0,
        iter_max: int = 100,
        damping: float = 2e-2,
        cg_max_iter: int = 100,
        cg_tol: float = 1e-12,
        use_truncated_cg: bool = True,
        diagnostics: bool = True,
        diagnostics_freq: int = 10,
        **kwargs,
    ):
        super().__init__(
            cost_fn,
            map,
            halt,
            print_cost,
            print_cost_freq,
            precision,
            ord_grad_u,
            ord_grad_theta,
            **kwargs,
        )

        self.name = "cg_newton"
        self.line_search = LineSearches[line_search_method]()

        self.iter_max = tf.Variable(iter_max, dtype=tf.int32)
        self.alpha_min = tf.Variable(alpha_min, dtype=self.precision)

        self.damping = tf.constant(damping, dtype=self.precision)
        self.cg_max_iter = cg_max_iter
        self.cg_tol = tf.constant(cg_tol, dtype=self.precision)
        self.use_truncated_cg = use_truncated_cg
        
        # Diagnostic settings
        self.diagnostics = diagnostics
        self.diagnostics_freq = diagnostics_freq

    def update_parameters(self, iter_max: int, damping: float) -> None:
        self.iter_max.assign(iter_max)
        self.damping = tf.cast(damping, self.precision)

    # ============================================================
    # Cost + gradient
    # ============================================================

    def _cost_and_grad(
        self, 
        inputs: tf.Tensor,
        cost_fn: Callable,
    ):
        with tf.GradientTape() as tape:
            U, V = self.map.get_UV(inputs)
            tape.watch((U, V))
            cost = cost_fn(U, V, inputs)

        grad_U, grad_V = tape.gradient(cost, (U, V))
        return cost, U, V, grad_U, grad_V

    # ============================================================
    # Hessian–vector product using backward-over-backward
    # ============================================================

    def _hvp(
        self,
        inputs: tf.Tensor,
        U: tf.Tensor,
        V: tf.Tensor,
        v_flat: tf.Tensor,
        cost_fn: Callable,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute (H + damping I) v using nested GradientTapes.
        """
        
        nU = tf.size(U)
        vU = tf.reshape(v_flat[:nU], tf.shape(U))
        vV = tf.reshape(v_flat[nU:], tf.shape(V))

        with tf.GradientTape() as outer_tape:
            outer_tape.watch((U, V))
            with tf.GradientTape() as inner_tape:
                inner_tape.watch((U, V))
                c = cost_fn(U, V, inputs)
            gU, gV = inner_tape.gradient(c, (U, V))
            
            gv = tf.reduce_sum(gU * vU) + tf.reduce_sum(gV * vV)
        
        HvU, HvV = outer_tape.gradient(gv, (U, V))

        Hv_flat = tf.concat(
            [tf.reshape(HvU, [-1]), tf.reshape(HvV, [-1])],
            axis=0,
        )

        return Hv_flat + damping * v_flat

    # ============================================================
    # DIAGNOSTICS: Lanczos algorithm for eigenvalue estimates
    # ============================================================

    def _lanczos_eigenvalues(
        self,
        inputs: tf.Tensor,
        U: tf.Tensor,
        V: tf.Tensor,
        cost_fn: Callable,
        damping: tf.Tensor,
        n_iter: int = 20,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Use Lanczos iteration to estimate extreme eigenvalues of H.
        Returns (lambda_min, lambda_max) estimates.
        
        This is matrix-free and only uses Hessian-vector products.
        """
        # Get problem dimension
        cost, _, _, grad_U, grad_V = self._cost_and_grad(inputs, cost_fn)
        g_flat = tf.concat([tf.reshape(grad_U, [-1]), tf.reshape(grad_V, [-1])], axis=0)
        n = tf.size(g_flat)
        
        # Initialize with random vector
        v = tf.random.normal(tf.shape(g_flat), dtype=self.precision)
        v = v / tf.norm(v)
        
        # Lanczos tridiagonal matrix
        alpha = tf.TensorArray(dtype=self.precision, size=n_iter, dynamic_size=False)
        beta = tf.TensorArray(dtype=self.precision, size=n_iter, dynamic_size=False)
        
        v_old = tf.zeros_like(v)
        beta_old = tf.constant(0.0, dtype=self.precision)
        
        for j in tf.range(n_iter):
            # Hessian-vector product
            w = self._hvp(inputs, U, V, v, cost_fn, damping)
            
            # Orthogonalize
            alpha_j = tf.reduce_sum(w * v)
            w = w - alpha_j * v - beta_old * v_old
            
            beta_j = tf.norm(w)
            
            alpha = alpha.write(j, alpha_j)
            beta = beta.write(j, beta_j)
            
            # Update
            v_old = v
            v = w / (beta_j + 1e-16)  # Avoid division by zero
            beta_old = beta_j
        
        # Build tridiagonal matrix
        alpha_vals = alpha.stack()
        beta_vals = beta.stack()
        
        # Estimate eigenvalues from tridiagonal (simple bounds)
        # lambda_max ≈ max(alpha + 2*beta)
        # lambda_min ≈ min(alpha - 2*beta)
        lambda_max = tf.reduce_max(alpha_vals + 2.0 * beta_vals)
        lambda_min = tf.reduce_min(alpha_vals - 2.0 * beta_vals)
        
        return lambda_min, lambda_max

    # ============================================================
    # DIAGNOSTICS: Rayleigh quotient on gradient direction
    # ============================================================

    def _gradient_curvature(
        self,
        inputs: tf.Tensor,
        U: tf.Tensor,
        V: tf.Tensor,
        g_flat: tf.Tensor,
        cost_fn: Callable,
        damping: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute g^T H g / g^T g (curvature along gradient direction).
        If this is negative or very small, we're in a flat or negative curvature region.
        """
        Hg = self._hvp(inputs, U, V, g_flat, cost_fn, damping)
        gHg = tf.reduce_sum(g_flat * Hg)
        gg = tf.reduce_sum(g_flat * g_flat)
        
        return gHg / (gg + 1e-16)

    # ============================================================
    # Truncated Linear Conjugate Gradient with Diagnostics
    # ============================================================

    def _cg_solve(
        self,
        inputs: tf.Tensor,
        U: tf.Tensor,
        V: tf.Tensor,
        b: tf.Tensor,
        cost_fn: Callable,
        damping: tf.Tensor,
        cg_max_iter: int,
        cg_tol: tf.Tensor,
        use_truncated: bool,
        verbose: bool = False,
    ) -> tf.Tensor:
        """
        Solve (H + damping I) x = b using truncated linear CG with diagnostics.
        """

        x = tf.zeros_like(b)
        r = b
        p = r
        rs = tf.reduce_sum(r * r)
        
        initial_residual = tf.sqrt(rs)
        
        neg_curv_detected = False

        for i in tf.range(cg_max_iter):
            # Early exit if converged
            if rs <= cg_tol:
                if verbose:
                    tf.print("  CG converged at iteration", i)
                break
                
            Ap = self._hvp(inputs, U, V, p, cost_fn, damping)
            pAp = tf.reduce_sum(p * Ap)
            
            # Diagnostics every few iterations
            if verbose and i % 5 == 0:
                rel_residual = tf.sqrt(rs) / (initial_residual + 1e-16)
                tf.print("  CG iter:", i, 
                        "| rel_res:", rel_residual, 
                        "| p^T Ap:", pAp)
            
            # Check for negative curvature (truncated CG)
            if use_truncated and pAp <= 0.0:
                if verbose:
                    tf.print("  NEGATIVE CURVATURE detected at CG iter", i, "| p^T Ap =", pAp)
                neg_curv_detected = True
                if i == 0:
                    x = r  # Use gradient direction
                break
            
            denom = pAp
            alpha = rs / denom
            
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = tf.reduce_sum(r * r)
            
            beta = rs_new / rs
            p = r + beta * p
            rs = rs_new

        # Final diagnostics
        if verbose:
            final_residual = tf.sqrt(rs) / (initial_residual + 1e-16)
            tf.print("  CG finished: iter =", i, "| final rel_res =", final_residual)
            if neg_curv_detected:
                tf.print("  WARNING: Negative curvature encountered")

        return x

    # ============================================================
    # Newton direction with diagnostics
    # ============================================================

    def _get_grad(
        self, 
        inputs: tf.Tensor,
        cost_fn: Callable,
        damping: tf.Tensor,
        cg_max_iter: int,
        cg_tol: tf.Tensor,
        use_truncated: bool,
        verbose: bool = False,
    ):
        """
        Compute cost, gradient, and Newton–CG step.
        """

        cost, U, V, grad_U, grad_V = self._cost_and_grad(inputs, cost_fn)

        g_flat = tf.concat(
            [tf.reshape(grad_U, [-1]), tf.reshape(grad_V, [-1])],
            axis=0,
        )


        step_flat = self._cg_solve(
            inputs, U, V, -g_flat, cost_fn, damping, cg_max_iter, cg_tol, use_truncated, verbose
        )

        nU = tf.size(U)
        step_U = tf.reshape(step_flat[:nU], tf.shape(U))
        step_V = tf.reshape(step_flat[nU:], tf.shape(V))

        return cost, U, V, g_flat, (grad_U, grad_V), (step_U, step_V)

    # ============================================================
    # Utilities
    # ============================================================

    def _force_descent(
        self, p_flat: tf.Tensor, grad_theta_flat: tf.Tensor, _: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        dot_gp = self._dot(grad_theta_flat, p_flat)
        return tf.cond(dot_gp >= 0.0, lambda: -grad_theta_flat, lambda: p_flat), None

    def _apply_step(
        self, theta_flat: tf.Tensor, alpha: tf.Tensor, p_flat: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        return theta_flat + alpha * p_flat, None

    @tf.function(reduce_retracing=True)
    def _dot(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        dtype = self.precision
        return tf.tensordot(tf.cast(a, dtype), tf.cast(b, dtype), axes=1)

    # ============================================================
    # Line search
    # ============================================================

    def _line_search(
        self, 
        theta_flat: tf.Tensor, 
        p_flat: tf.Tensor, 
        input: tf.Tensor,
        cost_fn: Callable,
        damping: tf.Tensor,
        cg_max_iter: int,
        cg_tol: tf.Tensor,
        use_truncated: bool,
    ) -> tf.Tensor:
        def eval_fn(alpha: tf.Tensor) -> ValueAndGradient:
            theta_backup = self.map.copy_theta(self.map.get_theta())
            theta_alpha, _ = self._apply_step(theta_flat, alpha, p_flat)

            self.map.set_theta(self.map.unflatten_theta(theta_alpha))
            f, _, _, _, _, grad = self._get_grad(input, cost_fn, damping, cg_max_iter, cg_tol, use_truncated, False)
            grad_flat = self.map.flatten_theta(grad)
            df = self._dot(grad_flat, p_flat)

            self.map.set_theta(theta_backup)
            return ValueAndGradient(x=alpha, f=f, df=df)

        return self.line_search.search(theta_flat, p_flat, eval_fn)

    # ============================================================
    # Main optimization loop
    # ============================================================

    @tf.function(jit_compile=False)
    def minimize_impl(self, inputs: tf.Tensor) -> tf.Tensor:
        first_batch = self.sampler(inputs)
        if first_batch.shape[0] != 1:
            raise NotImplementedError("Newton–CG requires a single batch.")

        input = first_batch[0, :, :, :, :]

        # Extract values that need to be passed as arguments
        cost_fn = self.cost_fn
        damping = self.damping
        cg_max_iter = self.cg_max_iter
        cg_tol = self.cg_tol
        use_truncated = self.use_truncated_cg
        diagnostics = self.diagnostics
        diag_freq = self.diagnostics_freq

        theta_flat = self.map.flatten_theta(self.map.get_theta())

        cost, U, V, g_flat, grad_u, step_theta = self._get_grad(
            input, cost_fn, damping, cg_max_iter, cg_tol, use_truncated, False
        )
        self._init_step_state(U, V, theta_flat)

        halt_status = tf.constant(HaltStatus.CONTINUE.value, dtype=tf.int32)
        iter_last = tf.constant(-1, dtype=tf.int32)
        costs = tf.TensorArray(dtype=cost.dtype, size=int(self.iter_max))

        for iter in tf.range(self.iter_max):

            # Run diagnostics at specified frequency
            run_diagnostics = tf.logical_and(
                diagnostics,
                tf.equal(tf.math.floormod(iter, diag_freq), 0)
            )

            if run_diagnostics:
                tf.print("\n========== DIAGNOSTICS AT ITERATION", iter, "==========")
                
                # 1. Eigenvalue estimates via Lanczos
                lambda_min, lambda_max = self._lanczos_eigenvalues(
                    input, U, V, cost_fn, damping, n_iter=20
                )
                tf.print("Estimated eigenvalues: λ_min =", lambda_min, ", λ_max =", lambda_max)
                
                # 2. Condition number
                cond_number = lambda_max / (tf.abs(lambda_min) + 1e-16)
                tf.print("Estimated condition number: κ(H) =", cond_number)
                
                # 3. Strong convexity check
                if lambda_min > 0.0:
                    tf.print("Status: STRONGLY CONVEX (λ_min > 0)")
                elif lambda_min > -1e-6:
                    tf.print("Status: NEARLY SINGULAR or FLAT (λ_min ≈ 0)")
                else:
                    tf.print("Status: INDEFINITE (λ_min < 0) - possible saddle point")
                
                # 4. Curvature along gradient direction
                grad_curv = self._gradient_curvature(input, U, V, g_flat, cost_fn, damping)
                tf.print("Curvature along gradient: g^T H g / ||g||^2 =", grad_curv)
                
                # 5. Gradient norm
                grad_norm = tf.norm(g_flat)
                tf.print("Gradient norm: ||g|| =", grad_norm)
                
                tf.print("=" * 60, "\n")

            # Solve Newton system with verbose output if diagnostics enabled
            cost, U, V, g_flat, grad_u, step_theta = self._get_grad(
                input, cost_fn, damping, cg_max_iter, cg_tol, use_truncated, run_diagnostics
            )

            p_flat = self.map.flatten_theta(step_theta)
            grad_flat = self.map.flatten_theta(grad_u)

            p_flat, _ = self._force_descent(p_flat, grad_flat, theta_flat)

            alpha = self._line_search(
                theta_flat, p_flat, input, cost_fn, damping, cg_max_iter, cg_tol, use_truncated
            )
            alpha = tf.maximum(alpha, tf.cast(self.alpha_min, alpha.dtype))

            theta_flat, _ = self._apply_step(theta_flat, alpha, p_flat)
            self.map.set_theta(self.map.unflatten_theta(theta_flat))

            costs = costs.write(iter, cost)

            U, V = self.map.get_UV(input)
            grad_u_norm, step_norm = self._get_grad_norm(grad_u, step_theta)
            self._update_step_state(
                iter, U, V, theta_flat, cost, grad_u_norm, step_norm
            )

            halt_status = self._check_stopping()
            self._update_display()

            iter_last = iter
            if tf.not_equal(halt_status, HaltStatus.CONTINUE.value):
                break
        
        tf.print("Final values for U and V", U, V)

        self._finalize_display(halt_status)
        return costs.stack()[: iter_last + 1]