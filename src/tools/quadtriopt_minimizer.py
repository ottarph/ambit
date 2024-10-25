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
        [ 0.5, 0.5],
        [ 0.0, 0.5],
        [ 0.5, 0.0],
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




def np_convex_global_minimizer_with_c_on_reference(c: np.ndarray) -> float:

    k = 4 * c[...,0] * c[...,2] - c[...,1]**2
    x = ( 2*c[...,2] * (-c[...,3]) - c[...,1] * (-c[...,4]) ) / k
    y = ( -c[...,1] * (-c[...,3]) + 2*c[...,0] * (-c[...,4]) ) / k

    vals = c[...,0] * x**2 + c[...,1] * x*y + \
            c[...,2] * y**2 + c[...,3] * x + \
            c[...,4] * y + c[...,5]
    
    valid = np.logical_and.reduce([0 <= x, 0 <= y, x + y <= 1])

    out = np.stack([
        np.where(valid, vals, np.inf),
        np.where(valid, x, np.nan),
        np.where(valid, y, np.nan),
    ], axis=-1)

    return out


def np_minimizer_on_edges_with_c_on_reference(uh: np.ndarray, c: np.ndarray) -> float:
    
    mins = np.zeros((*c.shape[:-1], 3))
    minimizers = np.zeros((*c.shape[:-1], 3, 2))

    # Need to check if solutions exist on each boundary
    
    # Edge 0->2, x = 0
    tc = np.where(c[...,[2]] <= 0, 
                                c * 0.0 + np.array([1.0, 1.0, 1.0, 1.0, -1.0, np.inf]),
                                c)
    y = -tc[...,4] / ( 2*tc[...,2] )
    y = np.fmax(np.zeros_like(y), y)
    y = np.fmin(np.ones_like(y), y)
    tmins = tc[...,2]*y**2 + tc[...,4]*y + tc[...,5]

    e_mins = np.stack([tmins, uh[...,0], uh[...,2]], axis=-1)
    ys = np.stack([y, np.zeros_like(y), np.ones_like(y)], axis=-1)
    min_inds = np.argmin(e_mins, axis=-1)
    corr_y = ys[np.arange(min_inds.shape[0]), min_inds]
    min_val = e_mins[np.arange(e_mins.shape[0]), min_inds]
    minimizers[...,0,:] = np.stack([np.zeros_like(corr_y), corr_y], axis=-1)
    mins[...,0] = min_val
    

    # Edge 0->1, y = 0
    tc = np.where(c[...,[0]] <= 0, 
                                c * 0.0 + np.array([1.0, 1.0, 1.0, -1.0, 1.0, np.inf]),
                                c)
    x = -tc[...,3] / ( 2*tc[...,0] )
    x = np.fmax(np.zeros_like(x), x)
    x = np.fmin(np.ones_like(x), x)
    tmins = tc[...,0]*x**2 + tc[...,3]*x + tc[...,5]

    e_mins = np.stack([tmins, uh[...,0], uh[...,1]], axis=-1)
    xs = np.stack([x, np.zeros_like(x), np.ones_like(x)], axis=-1)
    min_inds = np.argmin(e_mins, axis=-1)
    corr_x = xs[np.arange(min_inds.shape[0]), min_inds]
    min_val = e_mins[np.arange(e_mins.shape[0]), min_inds]
    minimizers[...,1,:] = np.stack([corr_x, np.zeros_like(corr_x)], axis=-1)
    mins[...,1] = min_val


    # Edge 1->2, x + y = 1
    tc = np.where(c[...,[0]] - c[...,[1]] + c[...,[2]] <= 0, 
                                c * 0.0 + np.array([1.0, 1.0, 1.0, 1.0, 1.0, np.inf]),
                                c)
    x = (-tc[...,1] + 2*tc[...,2] - tc[...,3] + tc[...,4]) / (2*tc[...,0] - 2*tc[...,1] + 2*tc[...,2])
    x = np.fmax(np.zeros_like(x), x)
    x = np.fmin(np.ones_like(x), x)
    tmins = tc[...,0]*x**2 + tc[...,1]*x*(1-x) + tc[...,2]*(1-x)**2 + \
          tc[...,3]*x + tc[...,4]*(1-x) + tc[...,5]

    e_mins = np.stack([tmins, uh[...,1], uh[...,2]], axis=-1)
    xs = np.stack([x, np.ones_like(x), np.zeros_like(x)], axis=-1)
    min_inds = np.argmin(e_mins, axis=-1)
    corr_x = xs[np.arange(min_inds.shape[0]), min_inds]
    min_val = e_mins[np.arange(e_mins.shape[0]), min_inds]
    minimizers[...,2,:] = np.stack([corr_x, 1 - corr_x], axis=-1)
    mins[...,2] = min_val

    # Return the minimum of the three edge-minimums

    min_inds = np.argmin(mins, axis=-1)
    corr_mins = mins[np.arange(mins.shape[0]), min_inds]
    corr_minimizers = minimizers[np.arange(mins.shape[0]), min_inds, :]

    out = np.stack([corr_mins, corr_minimizers[...,0], corr_minimizers[...,1]], axis=-1)

    return out


def minimizer_quadratic_on_reference(uh: np.ndarray) -> np.ndarray:

    c = np.einsum("ij,...j->...i", Kinv, uh)
    conv_c = np.where(4 * c[...,[0]] * c[...,[2]] - c[...,[1]]**2 <= 0,
                      0 * c + np.array([1.0, 0.0, 1.0, 0.0, 0.0, np.inf]),
                      c)

    conv_int_min_and_minimzer = np_convex_global_minimizer_with_c_on_reference(conv_c)
    edge_min_and_minimizer = np_minimizer_on_edges_with_c_on_reference(uh, c)

    out = np.where(conv_int_min_and_minimzer[...,[0]] < edge_min_and_minimizer[...,[0]], 
                   conv_int_min_and_minimzer, edge_min_and_minimizer)

    return out


def minimizer_quadratic_over_triangle(uh: np.ndarray, xh: np.ndarray | None = None) -> np.ndarray:
    
    original_length = len(uh.shape)
    if len(uh.shape) == 1:
        uh = uh[None,...]

    assert len(uh.shape) == 2

    if xh is None:
        minimizers = minimizer_quadratic_on_reference(uh)
        if original_length == 1:
            minimizers = minimizers[0,...]
        return minimizers
    
    assert len(xh.shape) == 3
    
    A = np.zeros((xh.shape[0], 2, 2), dtype=np.float64)
    b = xh[:,0,:]
    A[:,:,0] = xh[:,1,:] - b
    A[:,:,1] = xh[:,2,:] - b
    
    minimizers = minimizer_quadratic_on_reference(uh)

    np.einsum("bij,bj->bi", A, minimizers[:,1:], out=minimizers[:,1:])
    np.add(b, minimizers[:,1:], out=minimizers[:,1:])

    if original_length == 1:
        minimizers = minimizers[0,...]

    return minimizers






def main():

    import matplotlib.pyplot as plt

    import dolfinx as dfx
    from mpi4py.MPI import COMM_WORLD as comm



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
        minimum = mins[:,0]
        minimizer = mins[:,1:]

        plt.figure()
        plt.plot(cell_centers[k,0], cell_centers[k,1], 'o', label=f"{k=}")
        for l in range(4):
            plt.plot(minimizer[l,0], minimizer[l,1], 'x', label=f"{k=}, {l=}")

        for c in range(4):
            cell_x = V.tabulate_dof_coordinates().reshape(-1,6,3)[c,:,:2]
            plt.plot(cell_x[[0,1],0], cell_x[[0,1],1], 'k-', alpha=0.3, lw=0.4)
            plt.plot(cell_x[[1,2],0], cell_x[[1,2],1], 'k-', alpha=0.3, lw=0.4)
            plt.plot(cell_x[[2,0],0], cell_x[[2,0],1], 'k-', alpha=0.3, lw=0.4)

        plt.legend()
        plt.savefig(f"minimizer{k}.pdf")


    center = np.array([[0.2], [1.2], [0.0]])
    def interp(x):
            return np.sum((x - center)**2, axis=0)
    
    u.interpolate(interp)
    uh = u.x.array[:].reshape(-1,6)
    xh = V.tabulate_dof_coordinates().reshape(-1,6,3)[:,:3,:2]
    mins = minimizer_quadratic_over_triangle(uh, xh)
    minimum = mins[:,0]
    minimizer = mins[:,1:]

    plt.figure()
    plt.plot(center[0,0], center[1,0], 'o', label=f"{k=}")
    for l in range(4):
        plt.plot(minimizer[l,0], minimizer[l,1], 'x', label=f"{k=}, {l=}")

    for c in range(4):
        cell_x = V.tabulate_dof_coordinates().reshape(-1,6,3)[c,:,:2]
        plt.plot(cell_x[[0,1],0], cell_x[[0,1],1], 'k-', alpha=0.3, lw=0.4)
        plt.plot(cell_x[[1,2],0], cell_x[[1,2],1], 'k-', alpha=0.3, lw=0.4)
        plt.plot(cell_x[[2,0],0], cell_x[[2,0],1], 'k-', alpha=0.3, lw=0.4)

    plt.legend()

    plt.savefig("minimizer5.pdf")


    return


if __name__ == "__main__":
    main()



__all__ = [minimizer_quadratic_over_triangle]
