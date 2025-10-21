#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

from omegaconf import DictConfig
from typing import Any, Dict, Tuple
import tensorflow as tf

from igm.common.core import State
from igm.processes.iceflow.energy.energy import iceflow_energy


class SolverParams(tf.experimental.ExtensionType):
    Nx: int
    Ny: int
    batch_size: int
    staggered_grid: int
    print_cost: bool


def get_solver_params_args(
    cfg: DictConfig, Nx: int, Ny: int, batch_size: int
) -> Dict[str, Any]:

    cfg_solver = cfg.processes.iceflow.solver
    cfg_numerics = cfg.processes.iceflow.numerics

    return {
        "Nx": Nx,
        "Ny": Ny,
        "batch_size": batch_size,
        "staggered_grid": cfg_numerics.staggered_grid,
        "print_cost": cfg_solver.print_cost,
    }


def get_solver_bag(cfg: DictConfig, state: State) -> Dict[str, Any]:

    cfg_solver = cfg.processes.iceflow.solver

    fieldin = {name: getattr(state, name)[None, ...] for name in cfg_solver.fieldin}

    return {
        "fieldin": fieldin,
        "energy_components": state.iceflow.energy_components,
        "optimizer": state.optimizer,
        "nbit": cfg_solver.nbitmax,
        "vertical_discr": state.iceflow.vertical_discr,
    }


tf.config.optimizer.set_jit(True)


@tf.function(jit_compile=False)
def optimize_adam(
    bag: Dict[str, Any], U: tf.Tensor, V: tf.Tensor, parameters: SolverParams
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    emulator_cost_tensor = tf.TensorArray(dtype=tf.float32, size=bag["nbit"])

    for iteration in tf.range(bag["nbit"]):
        cost_emulator = 0.0

        with tf.GradientTape(persistent=True) as tape:

            nonstaggered_energy, staggered_energy = iceflow_energy(
                U[None, :, :, :],
                V[None, :, :, :],
                bag["fieldin"],
                bag["vertical_discr"],
                bag["energy_components"],
                parameters.staggered_grid,
                parameters.batch_size,
                parameters.Ny,
                parameters.Nx,
            )

            energy_mean_staggered = tf.reduce_mean(staggered_energy, axis=[1, 2, 3])
            energy_mean_nonstaggered = tf.reduce_mean(
                nonstaggered_energy, axis=[1, 2, 3]
            )

            total_energy = tf.reduce_sum(
                energy_mean_nonstaggered, axis=0
            ) + tf.reduce_sum(energy_mean_staggered, axis=0)
            cost_emulator += total_energy

        gradients = tape.gradient(total_energy, [U, V])

        bag["optimizer"].apply_gradients(zip(gradients, [U, V]))

        del tape

        if parameters.print_cost:
            tf.print("Iteration", iteration + 1, "/", bag["nbit"], end=" ")
            tf.print(": Cost =", cost_emulator)

        emulator_cost_tensor = emulator_cost_tensor.write(iteration, cost_emulator)

    return U, V, emulator_cost_tensor.stack()


# TODO: re-add L-BFGS


def solve_iceflow(cfg: DictConfig, state: State, init: bool = False) -> None:

    # Optimize and save cost
    bag = get_solver_bag(cfg, state)

    U = tf.Variable(state.U)
    V = tf.Variable(state.V)

    state.U, state.V, state.cost_solver = optimize_adam(
        bag, U, V, state.iceflow.solver_params
    )
