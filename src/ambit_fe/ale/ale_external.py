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


    def insert_correction(correction: np.ndarray, r: PETSc.Vec) -> None:
        r.array[:] -= correction # In residual form the correction has negative sign.
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
    


def find_local_sensors(sensors: np.ndarray, V: dfx.fem.FunctionSpace) -> tuple[np.ndarray, np.ndarray]:

    # comm = V.mesh.comm
    comm = MPI.COMM_WORLD

    local_found = np.zeros(len(sensors), dtype=np.int32)

    # print(f"{comm.rank = }, {V.dofmap.index_map.size_local = }, {V.tabulate_dof_coordinates().shape = }")


    cand_sensor_dofs = np.arange(V.dofmap.index_map.size_local)
    cand_sensor_x = V.tabulate_dof_coordinates()[cand_sensor_dofs,:sensors.shape[1]]

    diff = sensors[:,None,:] - cand_sensor_x[None,:,:]
    dist = np.linalg.norm(diff, axis=-1)
    closest = np.argmin(dist, axis=-1)
    closest_dist = np.min(dist, axis=-1)

    eps = 1e-8

    local_found[closest_dist < eps] = 1

    sensors_found = np.arange(len(sensors))[closest_dist < eps]
    index_where_found = cand_sensor_dofs[closest[closest_dist < eps]]

    global_sensors_found = comm.reduce(local_found, op=MPI.SUM, root=0)
    if comm.rank == 0:
        assert np.allclose(global_sensors_found, 1)

    # print(f"{comm.rank = }, {sensors_found.shape = }, {index_where_found.shape = }")
    # print(f"{comm.rank = }, {sensors_found = }, {index_where_found = }")


    pos_where_found = np.zeros((len(sensors), sensors.shape[1]))
    pos_where_found[sensors_found] = V.tabulate_dof_coordinates()[index_where_found,:sensors.shape[1]]

    global_pos_where_found = comm.reduce(pos_where_found, op=MPI.SUM, root=0)
    if comm.rank == 0:
        assert np.allclose(global_pos_where_found, sensors)


    return sensors_found, index_where_found


def add_my_hook_parallel(model_path: PathLike, V_d: fem.FunctionSpace, T: fem.FunctionSpace) -> callable:
    model_path = Path(model_path)

    # comm = V_d.mesh.comm
    comm = MPI.COMM_WORLD

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

    sensors_found, index_where_found = find_local_sensors(sensors, V_d)

    # Adjust for dofs being placed together in blocks, while find_local_sensors
    # works as if on scalar function spaces.
    local_branch_input_targets = np.zeros(sensors_found.shape[0]*2, dtype=np.int32)
    local_branch_input_targets[0::2] = sensors_found*2
    local_branch_input_targets[1::2] = sensors_found*2 + 1

    local_branch_input_sources = np.zeros(sensors_found.shape[0]*2, dtype=np.int32)
    local_branch_input_sources[0::2] = index_where_found*2
    local_branch_input_sources[1::2] = index_where_found*2 + 1

    # Vector to store the local branch inputs found.
    local_branch_inputs = np.zeros(sensors.shape[0]*2, dtype=np.float64)

    # Example: local_branch_inputs[local_branch_input_targets] = x[local_branch_input_sources]

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
        du.x.array[:] = 0.0
        
        # M.mult(t.x.petsc_vec, du.x.petsc_vec)
        M.mult(t.vector, du.vector)

        du.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        du.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        
        post_t_basis[p,:] = du.x.array[:]
        
    post_t_basis = post_t_basis[:,:V_d.dofmap.index_map.size_local*V_d.dofmap.bs]


    # Define functions to collect inputs and build corrections to residual
  
    @dfx.common.timed("collect_branch_inputs")
    def collect_branch_inputs(global_inputs: np.ndarray[np.float64] | None, local_inputs: np.ndarray[np.float64], 
                              local_input_targets: np.ndarray[np.int32], local_input_sources: np.ndarray[np.int32], 
                              x: PETSc.Vec) -> None:
        """
        Collects the branch network inputs from each rank and sends to root.
        """

        # comm = x.comm
        comm = MPI.COMM_WORLD

        if local_input_targets.shape[0] > 0:
            with x.localForm() as xl:
                local_inputs[local_input_targets] = xl[local_input_sources]

        comm.Reduce(local_inputs, global_inputs, op=MPI.SUM, root=0)

        return
    
    @dfx.common.timed("scatter_branch_outputs")
    def scatter_branch_outputs(global_outputs: np.ndarray[np.float64]):
        comm = MPI.COMM_WORLD

        comm.Bcast(global_outputs, root=0)

        return
    
    @dfx.common.timed("build_correction")
    def build_correction(correction_basis: np.ndarray, branch_output: np.ndarray) -> np.ndarray:
        return jnp.einsum("pi,p->i", correction_basis, branch_output)

    @dfx.common.timed("insert_correction")
    def insert_correction(correction: np.ndarray, r: PETSc.Vec) -> None:

        r.array[:] -= correction # In residual form the correction has negative sign.
        return
    

    # Define the hook

    @dfx.common.timed("hook_parallel")
    def hook(branch_network: Callable, 
             global_branch_inputs: np.ndarray[np.float64], global_branch_outputs: np.ndarray[np.float64],
             local_branch_inputs: np.ndarray[np.float64], 
             local_branch_input_targets: np.ndarray[np.float64], local_branch_input_sources: np.ndarray[np.float64], 
             local_correction_basis: np.ndarray[np.float64], time: float, x: PETSc.Vec, r: PETSc.Vec) -> None:

        comm = MPI.COMM_WORLD
        # comm = x.comm

        collect_branch_inputs(global_branch_inputs, local_branch_inputs, local_branch_input_targets, local_branch_input_sources, x)

        if comm.rank == 0:
            global_branch_outputs[:] = branch_network(global_branch_inputs)
        else:
            global_branch_outputs *= 0.0

        scatter_branch_outputs(global_branch_outputs)

        local_correction = build_correction(local_correction_basis, global_branch_outputs)

        insert_correction(local_correction, r)

        return
    
    
    if comm.rank == 0:
        global_branch_inputs = np.zeros(sensors.shape[0]*2, dtype=np.float64)
    else:
        global_branch_inputs = None
    global_branch_outputs = np.zeros(deeponet_size, dtype=np.float64)

    return jax.tree_util.Partial(hook, branch_mlp_comp, global_branch_inputs, global_branch_outputs,
                                 local_branch_inputs, local_branch_input_targets, local_branch_input_sources, 
                                 post_t_basis)
    

# ENDING: OTTAR HELLAN, 2024.10.07