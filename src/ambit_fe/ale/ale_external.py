# STARTING: OTTAR HELLAN, 2024.10.07

import dolfinx as dfx
import dolfinx.fem as fem
import dolfinx.fem.petsc
import ufl
from petsc4py import PETSc
from mpi4py import MPI

import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import json
from pathlib import Path
from os import PathLike

from typing import Callable

def mlp(Ws, bs, act, x):
    for W, b in zip(Ws[:-1], bs[:-1]):
        x = act(W @ x + b)
    return Ws[-1] @ x + bs[-1]


def add_my_hook(model_path: PathLike, V_d: fem.FunctionSpace, T: fem.FunctionSpace) -> callable:
    model_path = Path(model_path)

    # Load the model
    sensors = np.loadtxt(model_path / "sensor_points.txt")
    branch_params = [list(np.load(model_path / "branch_weights.npz").values()), list(np.load(model_path / "branch_biases.npz").values())]
    trunk_params = [list(np.load(model_path / "trunk_weights.npz").values()), list(np.load(model_path / "trunk_biases.npz").values())]
    
    branch_ws, branch_bs = branch_params
    trunk_ws, trunk_bs = trunk_params
    activation = json.load(open(model_path / "architecture.json", "r"))["activation"]
    activation = {"relu": jax.nn.relu, "tanh": jax.nn.tanh}[activation.lower()]

    deeponet_size = branch_bs[-1].shape[0]
    assert trunk_bs[-1].shape[0] == deeponet_size * (2*2)

    # Rescale first layer of branch network to work with millimeters.
    branch_ws[0] *= 1e-3

    branch_mlp = jax.tree_util.Partial(mlp, branch_ws, branch_bs, activation)
    trunk_mlp = jax.tree_util.Partial(mlp, trunk_ws, trunk_bs, activation)

    branch_mlp_comp = jax.jit(branch_mlp)
    branch_mlp_comp(jnp.zeros(branch_ws[0].shape[1]))

    # Rescale sensors to millimeter-scale.
    sensors *= 1e3

    # Find the sensors

    assert MPI.COMM_WORLD.size == 1, "Not implemented for parallel."

    sensor_cand_dofs = np.arange(V_d.tabulate_dof_coordinates().shape[0])
    sensor_cand = V_d.tabulate_dof_coordinates()[sensor_cand_dofs,:sensors.shape[1]]

    diff = sensors[:,None,:] - sensor_cand[None,:,:]
    dist = np.linalg.norm(diff, axis=-1)
    closest_subinds = np.argmin(dist, axis=-1)
    closest = sensor_cand_dofs[closest_subinds]

    assert np.allclose(V_d.tabulate_dof_coordinates()[closest,:sensors.shape[1]], sensors)

    local_sensor_inds = np.zeros(sensors.shape[0]*2, dtype=np.int32)
    local_sensor_inds[0::2] = closest*2
    local_sensor_inds[1::2] = closest*2 + 1


    # Get the eval points of trunk network

    trunk_eval_points = T.tabulate_dof_coordinates()[:,:2]
    # Rescale trunk inputs to meters
    trunk_eval_points *= 1e-3
    trunk_eval = jax.vmap(trunk_mlp)(trunk_eval_points)
    t_basis = trunk_eval.reshape(trunk_eval_points.shape[0], deeponet_size, 2*2)
    t_basis = t_basis.transpose(1, 0, 2)
    t_basis = t_basis.reshape(deeponet_size, trunk_eval_points.shape[0] * 2*2)

    

    # Build matrix to convert trunk outputs to right-hand-side basis functions

    T_dn = ufl.dot(ufl.TrialFunction(T), ufl.FacetNormal(T.mesh))
    m = ufl.inner(ufl.TrialFunction(T), ufl.grad(ufl.TestFunction(V_d))) * ufl.dx + ufl.inner(T_dn, ufl.TestFunction(V_d)) * ufl.ds

    m_form = fem.form(m)
    M = dolfinx.fem.petsc.assemble_matrix(m_form)
    M.assemble()


    # Fill array with right-hand-side basis functions

    t = fem.Function(T)
    du = fem.Function(V_d)

    post_t_basis = np.zeros((16, V_d.tabulate_dof_coordinates().shape[0] * 2))
    for p in range(deeponet_size):
        t.x.array[:] = t_basis[p,:]
        # M.mult(t.x.petsc_vec, du.x.petsc_vec)
        M.mult(t.vector, du.vector)
        post_t_basis[p,:] = du.x.array[:]
        


    # Define functions to collect inputs and build corrections to residual
  
    def collect_branch_inputs(local_sensor_inds: np.ndarray, x: PETSc.Vec) -> np.ndarray:
        """
        Collects the branch network inputs from each rank and sends to root.
        """
        assert MPI.COMM_WORLD.size == 1, "Not implemented for parallel."

        branch_inputs = x[local_sensor_inds]

        return branch_inputs
    

    def place_branch_outputs(branch_network: Callable, branch_inputs: np.ndarray, x: PETSc.Vec) -> np.ndarray:
        """
        Runs the branch network and spreads the branch outputs to non-root ranks.
        """
        assert MPI.COMM_WORLD.size == 1, "Not implemented for parallel."

        branch_output = branch_network(branch_inputs)

        return branch_output
    
    def build_correction(correction_basis: np.ndarray, branch_output: np.ndarray):
        return jnp.einsum("pi,p->i", correction_basis, branch_output)


    def insert_correction(correction: np.ndarray, x: PETSc.Vec) -> None:
        x.array[:] -= correction # In residual form the correction has negative sign.
        return
    

    # Define the hook

    @dfx.common.timed("hook")
    def hook(branch_network: Callable, local_sensor_inds: np.ndarray, correction_basis: np.ndarray, time: float, x: PETSc.Vec, r: PETSc.Vec) -> None:

        branch_inputs = collect_branch_inputs(local_sensor_inds, x)
        branch_outputs = place_branch_outputs(branch_network, branch_inputs, x)

        correction = build_correction(correction_basis, branch_outputs)

        insert_correction(correction, r)

        return
    
    return jax.tree_util.Partial(hook, branch_mlp_comp, local_sensor_inds, post_t_basis)
    
# ENDING: OTTAR HELLAN, 2024.10.07