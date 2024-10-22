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

    def __call__(self, fsi_problem: FSIProblem, N: int, t: float):

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

    def __call__(self, fsi_problem: FSIProblem, N: int, t: float):

        local_lift = dfx.fem.assemble_scalar(self.form)

        global_lift = self.comm.reduce(local_lift, op=MPI.SUM, root=0)

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},{global_lift}\n")

        return
    
class minimumDetFHook:
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
    
    @dfx.common.timed("minimumDetFHook")
    def __call__(self, fsi_problem: FSIProblem, N: int, t: float):

        self.j.interpolate(self.j_expr)

        jh = self.j.x.array[:self.ind_range].reshape(-1, 6)

        local_min_j = self.minimum_f(jh).min()

        global_min_j = self.comm.reduce(local_min_j, op=MPI.MIN, root=0)

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},{global_min_j}\n")

        return
    
class minimizerDetFHook:
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
    
    @dfx.common.timed("minimizerDetFHook")
    def __call__(self, fsi_problem: FSIProblem, N: int, t: float):

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

    def __call__(self, fsi_problem: FSIProblem, N: int, t: float):

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

    def __call__(self, fsi_problem: FSIProblem, N: int, t: float):

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