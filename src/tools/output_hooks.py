# STARTING: OTTAR HELLAN, 2024.10.08

import dolfinx as dfx
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

from os import PathLike

from ambit_fe.coupling.fsi_main import FSIProblem

class DragHook:

    def __init__(self, fsi_problem: FSIProblem, mu: float, save_path: PathLike, interface_tag: int = 1, obstacle_tag: int = 2,
                 quad_degree: int | None = None):


        fluid_ale_problem = fsi_problem.pbfa
        self.comm = fsi_problem.comm
        self.save_path = save_path

        io = fsi_problem.io
        iof = fsi_problem.iof


        self.ds = ufl.Measure("ds", domain=iof.mesh, subdomain_data=io.mt_b1_fluid, 
                            metadata={'quadrature_degree': io.quad_degree if quad_degree is None else quad_degree})

        ds_interface = self.ds(interface_tag)
        ds_obstacle = self.ds(obstacle_tag)

        e_x = dfx.fem.Constant(iof.mesh, (-1.0, 0.0))

        mu = dfx.fem.Constant(iof.mesh, mu)
        normal = ufl.FacetNormal(iof.mesh)

        u = fluid_ale_problem.pba.d
        v = fluid_ale_problem.pbf.v
        p = fluid_ale_problem.pbf.p

        Id = ufl.Identity(2)
        F = Id + ufl.grad(u)

        new_normal = ufl.dot(ufl.inv(F.T), normal)

        sigma_f_p = -p * Id
        sigma_f_v = mu * ( ufl.grad(v) * ufl.inv(F) + ufl.inv(F).T * ufl.grad(v).T )
        sigma_f = sigma_f_v + sigma_f_p

        form_obstacle = ufl.dot(ufl.dot(sigma_f, new_normal), e_x) * ufl.det(F) * ds_obstacle
        form_interface = ufl.dot(ufl.dot(sigma_f, new_normal), e_x) * ufl.det(F) * ds_interface

        self.form = dfx.fem.form(form_obstacle + form_interface)

        if self.comm.rank == 0:
            with open(self.save_path, "w") as f:
                f.write("time,drag\n")

        return

    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:

        u = fsi_problem.pbfa.pba.d
        v = fsi_problem.pbfa.pbf.v
        p = fsi_problem.pbfa.pbf.p

        local_drag = dfx.fem.assemble_scalar(self.form)

        global_drag = self.comm.reduce(local_drag, op=MPI.SUM, root=0)

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},{global_drag}\n")

        return
    
class LiftHook:

    def __init__(self, fsi_problem: FSIProblem, mu: float, save_path: PathLike, interface_tag: int = 1, obstacle_tag: int = 2,
                 quad_degree: int | None = None):


        fluid_ale_problem = fsi_problem.pbfa
        self.comm = fsi_problem.comm
        self.save_path = save_path

        io = fsi_problem.io
        iof = fsi_problem.iof


        self.ds = ufl.Measure("ds", domain=iof.mesh, subdomain_data=io.mt_b1_fluid, 
                            metadata={'quadrature_degree': io.quad_degree if quad_degree is None else quad_degree})

        ds_interface = self.ds(interface_tag)
        ds_obstacle = self.ds(obstacle_tag)

        e_y = dfx.fem.Constant(iof.mesh, (0.0, 1.0))

        mu = dfx.fem.Constant(iof.mesh, mu)
        normal = ufl.FacetNormal(iof.mesh)

        u = fluid_ale_problem.pba.d
        v = fluid_ale_problem.pbf.v
        p = fluid_ale_problem.pbf.p

        Id = ufl.Identity(2)
        F = Id + ufl.grad(u)

        new_normal = ufl.dot(ufl.inv(F.T), normal)

        sigma_f_p = -p * Id
        sigma_f_v = mu * ( ufl.grad(v) * ufl.inv(F) + ufl.inv(F).T * ufl.grad(v).T )
        sigma_f = sigma_f_v + sigma_f_p

        form_obstacle = ufl.dot(ufl.dot(sigma_f, new_normal), e_y) * ufl.det(F) * ds_obstacle
        form_interface = ufl.dot(ufl.dot(sigma_f, new_normal), e_y) * ufl.det(F) * ds_interface

        self.form = dfx.fem.form(form_obstacle + form_interface)

        if self.comm.rank == 0:
            with open(self.save_path, "w") as f:
                f.write("time,lift\n")

        return

    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:

        local_lift = dfx.fem.assemble_scalar(self.form)

        global_lift = self.comm.reduce(local_lift, op=MPI.SUM, root=0)

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},{global_lift}\n")

        return
    
class MinimumDetFHook:
    def __init__(self, fsi_problem: FSIProblem, save_path: PathLike):

        from tools.quadtriopt_minimum import minimum_quadratic_over_triangle
        self.minimum_f = minimum_quadratic_over_triangle

        self.save_path = save_path

        msh = fsi_problem.iof.mesh
        DG2 = dfx.fem.FunctionSpace(msh, ("DG", 2))
        self.comm = msh.comm

        self.ind_range = DG2.dofmap.index_map.size_local

        u = fsi_problem.pbfa.pba.d
        j_expr_ufl = ufl.det(ufl.Identity(msh.topology.dim) + ufl.grad(u))

        self.j_expr = dfx.fem.Expression(j_expr_ufl, DG2.element.interpolation_points())
        
        self.j = dfx.fem.Function(DG2)

        if self.comm.rank == 0:
            with open(self.save_path, "w") as f:
                f.write("time,min j\n")

        return
    
    @dfx.common.timed("MinimumDetFHook")
    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:

        self.j.interpolate(self.j_expr)

        jh = self.j.x.array[:self.ind_range].reshape(-1, 6)

        local_min_j = self.minimum_f(jh).min()

        global_min_j = self.comm.reduce(local_min_j, op=MPI.MIN, root=0)

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},{global_min_j}\n")

        return
    
class MinimizerDetFHook:
    def __init__(self, fsi_problem: FSIProblem, save_path: PathLike):

        from tools.quadtriopt_minimizer import minimizer_quadratic_over_triangle
        from functools import partial
        
        self.save_path = save_path

        msh = fsi_problem.iof.mesh
        DG2 = dfx.fem.FunctionSpace(msh, ("DG", 2))
        self.comm = msh.comm

        self.ind_range = DG2.dofmap.index_map.size_local

        xh = DG2.tabulate_dof_coordinates()[:self.ind_range,:].reshape(-1,6,3)[:,:3,:2]
        self.minimum_f = partial(minimizer_quadratic_over_triangle, xh=xh)

        u = fsi_problem.pbfa.pba.d
        j_expr_ufl = ufl.det(ufl.Identity(msh.topology.dim) + ufl.grad(u))

        self.j_expr = dfx.fem.Expression(j_expr_ufl, DG2.element.interpolation_points())
        
        self.j = dfx.fem.Function(DG2)

        if self.comm.rank == 0:
            self.rec_buffer = np.zeros((self.comm.size, 3))
        else:
            self.rec_buffer = None
        

        if self.comm.rank == 0:
            with open(self.save_path, "w") as f:
                f.write("time,min j,x,y\n")

        return
    
    @dfx.common.timed("MinimizerDetFHook")
    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:

        self.j.interpolate(self.j_expr)

        jh = self.j.x.array[:self.ind_range].reshape(-1, 6)

        cell_min_j = self.minimum_f(jh)
        min_ind = np.argmin(cell_min_j[:,0])
        message = cell_min_j[min_ind,:]

        self.comm.Gather(message, self.rec_buffer, root=0)

        if self.comm.rank == 0:

            global_min_ind = np.argmin(self.rec_buffer[:,0])
            global_min_j = self.rec_buffer[global_min_ind, 0]
            global_min_x = self.rec_buffer[global_min_ind, 1]
            global_min_y = self.rec_buffer[global_min_ind, 2]

            with open(self.save_path, "a") as f:
                f.write(f"{t},{global_min_j},{global_min_x},{global_min_y}\n")

        return
    
from tools.mesh_quality import scaled_jacobian_cellwise_impl, get_orientation_cellwise

class ScaledJacobianHook:

    def __init__(self, fsi_problem: FSIProblem, save_path: PathLike):

        self.save_path = save_path

        msh = fsi_problem.iof.mesh
        self.comm = msh.comm

        assert msh.topology.cell_name() == "triangle"
        vertex_slots = {"triangle": 3, "quadrilateral": 4}[msh.topology.cell_name()]

        self.u = fsi_problem.pbfa.pba.d
        V = self.u.function_space

        DG0 = dfx.fem.FunctionSpace(msh, ("DG", 0))
        

        self.num_local_cells = DG0.dofmap.index_map.size_local

        unique_offset_diffs = np.unique(np.diff(V.dofmap.list.offsets))
        assert len(unique_offset_diffs) == 1

        dmlist = V.dofmap.list.array.reshape(-1, unique_offset_diffs[0])
        
        self.local_cells = dmlist[:self.num_local_cells,:vertex_slots]
        self.local_x = V.tabulate_dof_coordinates()[self.local_cells.flatten(),:2].reshape(-1, vertex_slots, 2)

        self.local_orientation = get_orientation_cellwise(self.local_x)

        self.uh = np.zeros_like(self.local_x)

        del DG0

        if self.comm.rank == 0:
            with open(self.save_path, "w") as f:
                f.write("time,scaled jacobian [cells]\n")

        return
    
    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:
        
        self.uh[:,:,0] = self.u.x.array[2*self.local_cells.flatten()+0].reshape(self.num_local_cells, -1)
        self.uh[:,:,1] = self.u.x.array[2*self.local_cells.flatten()+1].reshape(self.num_local_cells, -1)

        local_sj = scaled_jacobian_cellwise_impl(self.local_x + self.uh, self.local_orientation)
        global_sj = self.comm.gather(local_sj)

        if self.comm.rank == 0:
            out_arr = np.concatenate(([[t]]+global_sj).reshape(-1,1))
            with open(self.save_path, "ab") as f:
                np.savetxt(f, out_arr, fmt="%.6f", delimiter=",")

        return
    
class MinScaledJacobianHook:

    def __init__(self, fsi_problem: FSIProblem, save_path: PathLike, include_internal_counter: bool = False,
                 write_time: bool = True):

        self.save_path = save_path

        msh = fsi_problem.iof.mesh
        self.comm = msh.comm

        assert msh.topology.cell_name() == "triangle"
        vertex_slots = {"triangle": 3, "quadrilateral": 4}[msh.topology.cell_name()]

        self.u = fsi_problem.pbfa.pba.d
        V = self.u.function_space

        DG0 = dfx.fem.FunctionSpace(msh, ("DG", 0))
        

        self.num_local_cells = DG0.dofmap.index_map.size_local

        unique_offset_diffs = np.unique(np.diff(V.dofmap.list.offsets))
        assert len(unique_offset_diffs) == 1

        dmlist = V.dofmap.list.array.reshape(-1, unique_offset_diffs[0])
        
        self.local_cells = dmlist[:self.num_local_cells,:vertex_slots]
        self.local_x = V.tabulate_dof_coordinates()[self.local_cells.flatten(),:2].reshape(-1, vertex_slots, 2)

        self.local_orientation = get_orientation_cellwise(self.local_x)

        self.uh = np.zeros_like(self.local_x)

        del DG0

        self.write_time = write_time
        self.include_internal_counter = include_internal_counter

        self.counter = -1

        if self.comm.rank == 0:
            with open(self.save_path, "w") as f:
                f.write("time,"*write_time + "solver iteration,"*include_internal_counter + "min scaled jacobian\n")

        return
    
    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:
        
        self.uh[:,:,0] = self.u.x.array[2*self.local_cells.flatten()+0].reshape(self.num_local_cells, -1)
        self.uh[:,:,1] = self.u.x.array[2*self.local_cells.flatten()+1].reshape(self.num_local_cells, -1)

        local_sj = scaled_jacobian_cellwise_impl(self.local_x + self.uh, self.local_orientation)
        local_min_sj = np.min(local_sj)
        global_min_sj = self.comm.reduce(local_min_sj, op=MPI.MIN, root=0)

        self.counter += 1

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},"*self.write_time+f"{self.counter},"*self.include_internal_counter+f"{global_min_sj}\n")

        return
    
class MatrixBlockNorm:

    def __init__(self, fsi_problem: FSIProblem, save_path: PathLike, block_indices: tuple[tuple[int, int]] | tuple[int, int],
                 write_time: bool, include_internal_counter: bool):
        self.save_path = save_path

        self.comm = fsi_problem.iof.mesh.comm

        self.block_indices = (block_indices, ) if isinstance(block_indices[0], int) else block_indices

        self.K_list = fsi_problem.K_list
        
        self.write_time = write_time
        self.include_internal_counter = include_internal_counter

        if self.comm.rank == 0:
            with open(self.save_path, "w") as f:
                f.write("time,"*write_time+"solver iteration,"*include_internal_counter+
                        ",".join((f"block{i}{j}" for (i,j) in self.block_indices))+"\n")

        return
    
    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:

        norms = []
        for i, j in self.block_indices:
            Jij = self.K_list[i][j]
            J_ij_norm = Jij.norm()
            norms.append(J_ij_norm)

        self.counter += 1

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},"*self.write_time+f"{self.counter},"*self.include_internal_counter+
                        ",".join(map(str, norms))+"\n")

        return
    
class ResetCounter:

    def __init__(self, hook, reset_value: int = -1):
        self.hook = hook
        self.reset_value = reset_value
        return

    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:
        self.hook.counter = self.reset_value
        return

class DragCornerHook:

    def __init__(self, fsi_problem: FSIProblem, mu: float, save_path: PathLike,
                 quad_degree: int | None = None):


        fluid_ale_problem = fsi_problem.pbfa
        self.comm = fsi_problem.comm
        self.save_path = save_path

        io = fsi_problem.io
        iof = fsi_problem.iof


        def finder(x):
            eps = 1e-3
            return np.logical_and.reduce([np.isclose(x[1], 190.0), x[0] <= 600+eps, x[0] >= 594.15-eps])
        
        found_ents = dfx.mesh.locate_entities_boundary(iof.mesh, 1, finder)
        marker_val = 13
        mt = dfx.mesh.meshtags(iof.mesh, 1, found_ents, marker_val)
        self.ds = ufl.Measure("ds", domain=iof.mesh, subdomain_id=marker_val, subdomain_data=mt, 
                            metadata={'quadrature_degree': io.quad_degree if quad_degree is None else quad_degree})

        e_x = dfx.fem.Constant(iof.mesh, (-1.0, 0.0))

        mu = dfx.fem.Constant(iof.mesh, mu)
        normal = ufl.FacetNormal(iof.mesh)

        u = fluid_ale_problem.pba.d
        v = fluid_ale_problem.pbf.v
        p = fluid_ale_problem.pbf.p

        Id = ufl.Identity(2)
        F = Id + ufl.grad(u)

        new_normal = ufl.dot(ufl.inv(F.T), normal)

        sigma_f_p = -p * Id
        sigma_f_v = mu * ( ufl.grad(v) * ufl.inv(F) + ufl.inv(F).T * ufl.grad(v).T )
        sigma_f = sigma_f_v + sigma_f_p

        form_ufl = ufl.dot(ufl.dot(sigma_f, new_normal), e_x) * ufl.det(F) * self.ds

        self.form = dfx.fem.form(form_ufl)

        if self.comm.rank == 0:
            with open(self.save_path, "w") as f:
                f.write("time,drag\n")

        return

    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:

        u = fsi_problem.pbfa.pba.d
        v = fsi_problem.pbfa.pbf.v
        p = fsi_problem.pbfa.pbf.p

        local_drag = dfx.fem.assemble_scalar(self.form)

        global_drag = self.comm.reduce(local_drag, op=MPI.SUM, root=0)

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},{global_drag}\n")

        return
    
class DetFCornerHook:

    def __init__(self, fsi_problem: FSIProblem, mu: float, save_path: PathLike,
                 quad_degree: int | None = None):


        fluid_ale_problem = fsi_problem.pbfa
        self.comm = fsi_problem.comm
        self.save_path = save_path

        io = fsi_problem.io
        iof = fsi_problem.iof


        def finder(x):
            eps = 1e-3
            return np.logical_and.reduce([np.isclose(x[1], 190.0), x[0] <= 600+eps, x[0] >= 594.15-eps])
        
        found_ents = dfx.mesh.locate_entities_boundary(iof.mesh, 1, finder)
        marker_val = 13
        mt = dfx.mesh.meshtags(iof.mesh, 1, found_ents, marker_val)
        self.ds = ufl.Measure("ds", domain=iof.mesh, subdomain_id=marker_val, subdomain_data=mt, 
                            metadata={'quadrature_degree': io.quad_degree if quad_degree is None else quad_degree})



        u = fluid_ale_problem.pba.d


        Id = ufl.Identity(2)
        F = Id + ufl.grad(u)
        J = ufl.det(F)

        form_ufl = J * self.ds

        self.form = dfx.fem.form(form_ufl)

        if self.comm.rank == 0:
            with open(self.save_path, "w") as f:
                f.write("time,drag\n")

        return

    def __call__(self, fsi_problem: FSIProblem, N: int, t: float) -> None:

        u = fsi_problem.pbfa.pba.d
        v = fsi_problem.pbfa.pbf.v
        p = fsi_problem.pbfa.pbf.p

        local_drag = dfx.fem.assemble_scalar(self.form)

        global_drag = self.comm.reduce(local_drag, op=MPI.SUM, root=0)

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},{global_drag}\n")

        return

# ENDING: OTTAR HELLAN, 2024.10.08