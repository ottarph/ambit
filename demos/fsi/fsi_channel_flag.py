#!/usr/bin/env python3

"""
FSI of elastic flag in channel (2D) (Turek benchmark): Q2-Q1 Taylor-Hood elements for fluid and Q2 elements for solid
Reference solution: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/fsi_benchmark/fsi_reference.html
"""

import ambit_fe
import numpy as np
from pathlib import Path


def main():

    basepath = str(Path(__file__).parent.absolute())

    # reads in restart step from the command line
    try: import sys; restart_step = int(sys.argv[1])
    except: restart_step = 0

    restart_step = 3000
    
    case = 'FSI2' # 'FSI2', 'FSI3'

    # cases from Tab. 12, Turek et al. 2006
    if case=='FSI2':
        # solid
        mu_s = 0.5e3 # kPa
        nu_s = 0.4
        rho0_s = 10.0e-6 # kg/mm^3
        # fluid
        mu_f = 1.0e-3 # kPa
        rho_f = 1.0e-6 # kg/mm^3
        # inflow vel
        Ubar = 1e3 # mm/s
        # max simulation time until periodic
        maxtime = 15.0
        maxtime = 14.0
        dt_ref = 0.0005
        dt_large = 0.004
    elif case=='FSI3':
        # solid
        mu_s = 2.0e3 # kPa
        nu_s = 0.4
        rho0_s = 1.0e-6 # kg/mm^3
        # fluid
        mu_f = 1.0e-3 # kPa
        rho_f = 1.0e-6 # kg/mm^3
        # inflow vel
        Ubar = 2e3 # mm/s
        # max simulation time until periodic
        maxtime = 7.5
        dt_ref = 0.00025
        dt_large = 0.002
    else:
        raise ValueError("Unknown case.")

    # dt_ref is the time step used to compute the reference solution of the original benchmark, cf. link above
    # dt_ref leads to 30000, dt_large to 3750 time steps in both cases
    # dt = dt_ref
    dt = dt_large

    """
    Parameters for input/output
    """
    IO_PARAMS            = {# problem type 'fsi': fluid-solid interaction
                            'problem_type'          : 'fsi',
                            'USE_OLD_DOLFINX_MIXED_BRANCH' : True,
                            # at which step frequency to write results
                            'write_results_every'   : 1,
                            'write_restart_every'   : 0,
                            'restart_step'          : restart_step,
                            # where to write the output to
                            'output_path'           : basepath+'/tmp_tri/16proc_mqtest',
                            # 'output_path'           : basepath+'/tmp_exp',
                            # 'mesh_domain'           : basepath+'/input/channel-flag_domain.xdmf',
                            # 'mesh_boundary'         : basepath+'/input/channel-flag_boundary.xdmf',
                            'mesh_domain': basepath+'/input/tri_cell_mesh.xdmf',
                            'mesh_boundary': basepath+'/input/tri_facet_mesh.xdmf',
                            # 'results_to_write'      : [['displacement','velocity'], [['fluiddisplacement','velocity','pressure'],['aledisplacement','alevelocity']]],
                            # 'results_to_write'      : [[], [['velocity','pressure'],['aledisplacement']]],
                            'results_to_write'      : [[], [[],[]]],
                            'domain_ids_solid'      : [1], 
                            'domain_ids_fluid'      : [2],
                            # 'surface_ids_interface' : [1],
                            'surface_ids_interface' : [11],
                            # 'gridname_domain': "Grid", # To work with my triangle mesh
                            # 'gridname_boundary': "Facet tags", # To work with my triangle mesh
                            # 'mesh_dim': 2,
                            # 'meshfile_format': 'XDMF',
                            # 'meshfile_type': 'ASCII',
                            'simname'               : 'fsi_channel_flag_turek_'+case}

    """
    Parameters for the linear and nonlinear solution schemes
    """
    SOLVER_PARAMS        = {'solve_type'            : 'direct',
                            'direct_solver'         : 'mumps',
                            # residual and increment tolerances
                            'tol_res'               : [1e-8,1e-8,1e-8,1e-8,1e-3], # solid-mom,fluid-mom,fluid-cont,FSI-coup,ALE-mom
                            'tol_inc'               : [1e-0,1e-0,1e-0,1e10,1e-0]} # du,dv,dp,dlm,dd

    """
    Parameters for the solid mechanics time integration scheme, plus the global time parameters
    """
    TIME_PARAMS_SOLID    = {'maxtime'               : maxtime,
                            'dt'                    : dt,
                            'timint'                : 'genalpha', # Generalized-alpha time-integration scheme (Chung and Hulbert 1993)
                            'rho_inf_genalpha'      : 1.0, # spectral radius of Gen-alpha: 1.0 (= no high-freq. damping) yields alpha_m = alpha_f = 0.5, beta = 0.25, gamma = 0.5
                            # how to evaluat nonlinear terms f(x) in the midpoint time-integration scheme:
                            # trapezoidal: theta * f(x_{n+1}) + (1-theta) * f(x_{n})
                            # midpoint:    f(theta*x_{n+1} + (1-theta)*x_{n})
                            'eval_nonlin_terms'     : 'midpoint'} # trapezoidal, midpoint

    """
    Parameters for the fluid mechanics time integration scheme, plus the global time parameters
    """
    TIME_PARAMS_FLUID    = {'maxtime'               : maxtime,
                            'dt'                    : dt,
                            'timint'                : 'genalpha', # Generalized-alpha time-integration scheme (Jansen et al. 2000)
                            'rho_inf_genalpha'      : 1.0, # spectral radius of Gen-alpha: 1.0 (= no high-freq. damping) yields alpha_m = alpha_f = 0.5, gamma = 0.5
                            # how to evaluate nonlinear terms f(x) in the midpoint time-integration scheme:
                            # trapezoidal: theta * f(x_{n+1}) + (1-theta) * f(x_{n})
                            # midpoint:    f(theta*x_{n+1} + (1-theta)*x_{n})
                            'eval_nonlin_terms'     : 'midpoint'} # trapezoidal, midpoint

    """
    Finite element parameters for solid: Taylor-Hood space
    """
    FEM_PARAMS_SOLID     = {'order_disp'            : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5,
                            'incompressible_2field' : False}

    """
    Finite element parameters for fluid: Taylor-Hood space
    """
    FEM_PARAMS_FLUID     = {'order_vel'             : 2,
                            'order_pres'            : 1,
                            'quad_degree'           : 5}
    
    """
    Finite element parameters for ALE
    """
    FEM_PARAMS_ALE       = {'order_disp'            : 2,
                            'quad_degree'           : 5}
    
    """
    FSI coupling parameters
    """
    # COUPLING_PARAMS      = {'coupling_fluid_ale'    : [{'surface_ids' : [1], 'type' : 'strong_dirichlet'}],
    COUPLING_PARAMS      = {'coupling_fluid_ale'    : [{'surface_ids' : [11], 'type' : 'strong_dirichlet'}],
                            'fsi_governing_type'    : 'solid_governed', # solid_governed, fluid_governed
                            'zero_lm_boundary'      : False} # TODO: Seems to select the wrong dofs on LM mesh! Do not turn on!


    # solid material: St.-Venant Kirchhoff
    MATERIALS_SOLID      = {'MAT1' : {'stvenantkirchhoff' : {'Emod' : 2.*mu_s*(1.+nu_s), 'nu' : nu_s},
                                      'inertia'           : {'rho0' : rho0_s}}} 

    # fluid material: standard Newtonian fluid
    MATERIALS_FLUID      = {'MAT1' : {'newtonian' : {'mu' : mu_f},
                                      'inertia'   : {'rho' : rho_f}}}
    
    # nonlinear material for domain motion problem: This has proved superior to the linear elastic model for large mesh deformations
    # MATERIALS_ALE        = {'MAT1' : {'exponential' : {'a_0' : 1.0, 'b_0' : 10.0, 'kappa' : 1e2}}}
    # MATERIALS_ALE        = {'MAT1' : {'weak_form_don' : {"model_path": basepath+"/input/models/wfambit002/model"}}}
    MATERIALS_ALE        = {'MAT1' : {'weak_form_don' : {"model_path": basepath+"/input/models/wf003fmcg1/model"}}}


    """
    User expression, here a spatially varying time-controlled inflow: always need a class variable self.t and an evaluate(self, x)
    with the only argument being the spatial coordinates x
    """
    class expression1:
        def __init__(self):
            
            self.t = 0.0

            self.t_ramp = 2.0
            
            self.H = 0.41e3 # channel height
            self.Ubar = Ubar

        def evaluate(self, x):
            
            vel_inflow_y = 1.5*self.Ubar*( x[1]*(self.H-x[1])/((self.H/2.)**2.) )

            val_t = vel_inflow_y * 0.5*(1.-np.cos(np.pi*self.t/self.t_ramp)) * (self.t < self.t_ramp) + vel_inflow_y * (self.t >= self.t_ramp)

            return ( np.full(x.shape[1], val_t),
                     np.full(x.shape[1], 0.0) )


    """
    Boundary conditions
    """
    # BC_DICT_SOLID        = { 'dirichlet' : [{'id' : [6], 'dir' : 'all', 'val' : 0.}] }
    BC_DICT_SOLID        = { 'dirichlet' : [{'id' : [25], 'dir' : 'all', 'val' : 0.}] }

    # BC_DICT_FLUID        = { 'dirichlet' : [{'id' : [4], 'dir' : 'all', 'expression' : expression1},
    #                                         {'id' : [2,3], 'dir' : 'all', 'val' : 0.}] }
    BC_DICT_FLUID        = { 'dirichlet' : [{'id' : [22], 'dir' : 'all', 'expression' : expression1},
                                            {'id' : [21,24], 'dir' : 'all', 'val' : 0.}] }

    # BC_DICT_ALE          = { 'dirichlet' : [{'id' : [2,3,4,5], 'dir' : 'all', 'val' : 0.}] }
    BC_DICT_ALE          = { 'dirichlet' : [{'id' : [21,22,23,24], 'dir' : 'all', 'val' : 0.}] }



    # Pass parameters to Ambit to set up the problem
    problem = ambit_fe.ambit_main.Ambit(IO_PARAMS, [TIME_PARAMS_SOLID, TIME_PARAMS_FLUID], SOLVER_PARAMS, [FEM_PARAMS_SOLID, FEM_PARAMS_FLUID, FEM_PARAMS_ALE], [MATERIALS_SOLID, MATERIALS_FLUID, MATERIALS_ALE], [BC_DICT_SOLID, BC_DICT_FLUID, BC_DICT_ALE], coupling_params=COUPLING_PARAMS)

    from tools.output_hooks import DragHook, LiftHook, DragCornerHook, MinimumDetFHook, MinimizerDetFHook, ScaledJacobianHook

    qoi_base_path = IO_PARAMS["output_path"]+'/results_'+IO_PARAMS["simname"]+f'_r{restart_step}'*(restart_step>0)
    problem.mp.output_hooks = [
        DragHook(problem.mp, mu_f, qoi_base_path+'_drag.txt', interface_tag=11, obstacle_tag=21),
        LiftHook(problem.mp, mu_f, qoi_base_path+'_lift.txt', interface_tag=11, obstacle_tag=21),
        DragCornerHook(problem.mp, mu_f, qoi_base_path+'_drag_corner.txt'),
        MinimizerDetFHook(problem.mp, qoi_base_path+'_minimizerDetF.txt'),
        # DragHook(problem.mp, mu_f, qoi_base_path+'_drag_q1.txt', interface_tag=11, obstacle_tag=21, quad_degree=1),
        # LiftHook(problem.mp, mu_f, qoi_base_path+'_lift_q1.txt', interface_tag=11, obstacle_tag=21, quad_degree=1),
        # DragCornerHook(problem.mp, mu_f, qoi_base_path+'_drag_corner_q1.txt', quad_degree=1),
        # DragHook(problem.mp, mu_f, qoi_base_path+'_drag_q3.txt', interface_tag=11, obstacle_tag=21, quad_degree=3),
        # LiftHook(problem.mp, mu_f, qoi_base_path+'_lift_q3.txt', interface_tag=11, obstacle_tag=21, quad_degree=3),
        # DragCornerHook(problem.mp, mu_f, qoi_base_path+'_drag_corner_q3.txt', quad_degree=3),
    ]

    from tools.output_hooks import MinScaledJacobianHook, ResetCounter, MatrixBlockNorm
    problem.mp.pbfa.pba.residual_assembly_hooks.extend([
        MinScaledJacobianHook(problem.mp, qoi_base_path+'_newton_iter_min_scaled_jacobian.txt',
                                    include_internal_counter=True, write_time=False),
        MatrixBlockNorm(problem.mp, problem.ms, qoi_base_path+'_newton_iter_blocknorms.txt', 
                        ((0,0),               (0,3), 
                                (1,1), (1,2), (1,3), (1,4), 
                                (2,1),               (2,4), # (2,2) is nonzero only when using stabilization for pressure
                         (3,0), (3,1),
                                (4,1),               (4,4)),
                        write_time=False, include_internal_counter=True)
        ])

    problem.mp.output_hooks.extend([
        ResetCounter(problem.mp.pbfa.pba.residual_assembly_hooks[-1]),
        ResetCounter(problem.mp.pbfa.pba.residual_assembly_hooks[-2]),
    ])

    # Call the Ambit solver to solve the problem
    problem.solve_problem()

    import dolfinx as dfx
    from mpi4py.MPI import COMM_WORLD
    dfx.common.list_timings(COMM_WORLD, [dfx.common.TimingType.wall], reduction=dfx.common.Reduction.max)


if __name__ == "__main__":

    main()
