import numpy as np

import numpy as np


"""
    The problem is set up like this:

    f(x,y) = a x_1^2 + b x_1 x_2 + c x_2^2 + d x_1 + e x_2 + f
           = C^T X, where
         C = [a, b, c, d, e, f],
         X = [x_1^2, x_1 x_2, x_2^2, x_1, x_2, 1]
           
    f is evaluated at the dolfinx DG2 dof locations,
    T = np.array([
        [ 0.0, 0.0],
        [ 1.0, 0.0],
        [ 0.0, 1.0],
        [ 0.5, 0.0],
        [ 0.0, 0.5],
        [ 0.5, 0.5],
    ])
    and
    U = f(T) = np.array([
        f( 0.0, 0.0),
        f( 1.0, 0.0),
        f( 0.0, 1.0),
        f( 0.5, 0.5),
        f( 0.0, 0.5),
        f( 0.5, 0.0),
    ])

    Then, K C = U, where

    K = K = np.array([
        [     0,     0,     0,     0,     0,     1],
        [     1,     0,     0,     1,     0,     1],
        [     0,     0,     1,     0,     1,     1],
        [   1/4,   1/4,   1/4,   1/2,   1/2,     1],
        [     0,     0,   1/4,     0,   1/2,     1],
        [   1/4,     0,     0,   1/2,     0,     1],
    ])

    giving C = K^{-1} U.

"""
# T = np.array([
#     [ 0.0, 0.0],
#     [ 1.0, 0.0],
#     [ 0.0, 1.0],
#     [ 0.5, 0.5],
#     [ 0.0, 0.5],
#     [ 0.5, 0.0],
# ])

K = np.array([
    [       0,       0,       0,       0,       0,       1],
    [       1,       0,       0,       1,       0,       1],
    [       0,       0,       1,       0,       1,       1],
    [     1/4,     1/4,     1/4,     1/2,     1/2,       1],
    [       0,       0,     1/4,       0,     1/2,       1],
    [     1/4,       0,       0,     1/2,       0,       1],
])
Kinv = np.linalg.inv(K)




def np_convex_global_minimum_with_c(c: np.ndarray) -> float:

    k = 4 * c[...,0] * c[...,2] - c[...,1]**2
    x = ( 2*c[...,2] * (-c[...,3]) - c[...,1] * (-c[...,4]) ) / k
    y = ( -c[...,1] * (-c[...,3]) + 2*c[...,0] * (-c[...,4]) ) / k

    vals = c[...,0] * x**2 + c[...,1] * x*y + \
            c[...,2] * y**2 + c[...,3] * x + \
            c[...,4] * y + c[...,5]

    return np.where(np.logical_and.reduce([0 <= x, 0 <= y, x + y <= 1]), 
                    vals,
                    np.inf)



def np_minimum_on_edges_with_c(uh: np.ndarray, c: np.ndarray) -> float:
    

    mins = np.zeros((*c.shape[:-1], 3))

    # Need to check if solutions exist on each boundary
    
    # Edge 0->2, x = 0
    tc = np.where(c[...,[2]] <= 0, 
                                c * 0.0 + np.array([1.0, 1.0, 1.0, 1.0, -1.0, np.inf]),
                                c)
    y = -tc[...,4] / ( 2*tc[...,2] )
    y = np.fmax(np.zeros_like(y), y)
    y = np.fmin(np.ones_like(y), y)
    tmins = tc[...,2]*y**2 + tc[...,4]*y + tc[...,5]
    smins = np.min(uh[...,[0,2]], axis=-1)
    mins[...,0] = np.fmin(tmins, smins)

    # Edge 0->1, y = 0
    tc = np.where(c[...,[0]] <= 0, 
                                c * 0.0 + np.array([1.0, 1.0, 1.0, -1.0, 1.0, np.inf]),
                                c)
    x = -tc[...,3] / ( 2*tc[...,0] )
    x = np.fmax(np.zeros_like(x), x)
    x = np.fmin(np.ones_like(x), x)
    tmins = tc[...,0]*x**2 + tc[...,3]*x + tc[...,5]
    smins = np.min(uh[...,[0,1]], axis=-1)
    mins[...,1] = np.fmin(tmins, smins)


    # Edge 1->2, x + y = 1
    tc = np.where(c[...,[0]] - c[...,[1]] + c[...,[2]] <= 0, 
                                c * 0.0 + np.array([1.0, 1.0, 1.0, 1.0, 1.0, np.inf]),
                                c)
    x = (-tc[...,1] + 2*tc[...,2] - tc[...,3] + tc[...,4]) / (2*tc[...,0] - 2*tc[...,1] + 2*tc[...,2])
    x = np.fmax(np.zeros_like(x), x)
    x = np.fmin(np.ones_like(x), x)
    tmins = tc[...,0]*x**2 + tc[...,1]*x*(1-x) + tc[...,2]*(1-x)**2 + \
          tc[...,3]*x + tc[...,4]*(1-x) + tc[...,5]
    smins = np.min(uh[...,[1,2]], axis=-1)
    mins[...,2] = np.fmin(tmins, smins)

    return np.min(mins, axis=-1)


def minimum_quadratic_over_triangle(uh: np.ndarray) -> np.ndarray:

    c = np.einsum("ij,...j->...i", Kinv, uh)

    conv_c = np.where(4 * c[...,[0]] * c[...,[2]] - c[...,[1]]**2 <= 0,
                      0 * c + np.array([1.0, 0.0, 1.0, 0.0, 0.0, np.inf]),
                      c)

    conv_int_mins = np_convex_global_minimum_with_c(conv_c)
    edge_mins = np_minimum_on_edges_with_c(uh, c)

    return np.fmin(conv_int_mins, edge_mins)


def main():

    
    import matplotlib.pyplot as plt

    import dolfinx as dfx
    from mpi4py.MPI import COMM_WORLD as comm

    from tools.quadtriopt_minimizer import minimizer_quadratic_over_triangle



    fmsh = dfx.mesh.create_unit_square(comm, 1, 1, cell_type=dfx.mesh.CellType.triangle)
    msh, *_ = dfx.mesh.create_submesh(fmsh, 2, np.array([0], dtype=np.int32))
    msh.geometry.x[2,[0,1]] = [0, 1]

    DG2 = dfx.fem.FunctionSpace(msh, ("DG", 2))
    u = dfx.fem.Function(DG2)

    with np.printoptions(precision=4, suppress=True):
        print(f"{msh.geometry.x[:] = }")
        print(f"{DG2.tabulate_dof_coordinates()[:,:2] = }")


    msh = dfx.mesh.create_unit_square(comm, 2, 1, cell_type=dfx.mesh.CellType.triangle)
    DG0 = dfx.fem.FunctionSpace(msh, ("DG", 0))
    cell_centers = DG0.tabulate_dof_coordinates().reshape(-1,3,1)

    V = dfx.fem.FunctionSpace(msh, ("DG", 2))
    u = dfx.fem.Function(V)


    
    scals = [1.0, 1.0, 1.0, 1.0]
    adds = [0.4, 1.3, 2.5, 3.0]
    for k in range(len(scals)):
        u.x.array[:] = 0.0
        def interp(x):
            return np.sum((x - cell_centers[k])**2, axis=0)*scals[k] + adds[k]
        u.interpolate(interp)
        uh = u.x.array[:].reshape(-1,6)
        xh = V.tabulate_dof_coordinates().reshape(-1,6,3)[:,:3,:2]
        mins = minimizer_quadratic_over_triangle(uh, xh)
        minimum = minimum_quadratic_over_triangle(uh)

        assert np.allclose(minimum, mins[:,0])
        assert np.allclose(minimum.min(), adds[k])



    center = np.array([[0.2], [1.2], [0.0]])
    def interp(x):
            return np.sum((x - center)**2, axis=0)
    
    u.interpolate(interp)
    uh = u.x.array[:].reshape(-1,6)
    xh = V.tabulate_dof_coordinates().reshape(-1,6,3)[:,:3,:2]
    mins = minimizer_quadratic_over_triangle(uh, xh)
    minimum = minimum_quadratic_over_triangle(uh)

    assert np.allclose(minimum, mins[:,0])

    return


if __name__ == "__main__":
    main()



__all__ = [minimum_quadratic_over_triangle]
