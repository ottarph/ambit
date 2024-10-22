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

        # for f in [u, v, p]:
        #     f.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        #     f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

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

        # for f in [u, v, p]:
        #     f.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        #     f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

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

        # for f in [u, v, p]:
        #     f.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        #     f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        local_drag = dfx.fem.assemble_scalar(self.form)

        global_drag = self.comm.reduce(local_drag, op=MPI.SUM, root=0)

        if self.comm.rank == 0:
            with open(self.save_path, "a") as f:
                f.write(f"{t},{global_drag}\n")

        return

# ENDING: OTTAR HELLAN, 2024.10.08