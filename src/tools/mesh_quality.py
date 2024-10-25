import numpy as np

def get_orientation(cells: np.ndarray, x: np.ndarray) -> np.ndarray:

    if cells.shape[0] == 3:
        cells = cells.T
    return np.sign(scaled_jacobian_impl(cells, x))

def get_orientation_cellwise(x: np.ndarray) -> np.ndarray:

    return np.sign(scaled_jacobian_cellwise_impl(x))

def scaled_jacobian_impl(cells: np.ndarray, z: np.ndarray, orientation: float | np.ndarray = 1.0) -> np.ndarray:

    edges = z[...,cells[:,[1,2,0]],:] - z[...,cells[:,[0,1,2]],:]
    jacobian = edges[...,0,0] * edges[...,1,1] - edges[...,0,1] * edges[...,1,0]

    edge_lengths = np.linalg.norm(edges, axis=-1, ord=2)

    max_edge_length_product = np.max(
        edge_lengths[...,[0,1,2]] * edge_lengths[...,[1,2,0]]
    , axis=-1)

    scaled_jacobian = 2.0 / 3.0**0.5 * jacobian / max_edge_length_product

    return scaled_jacobian * orientation


def scaled_jacobian_cellwise_impl(z: np.ndarray, orientation: float | np.ndarray = 1.0) -> np.ndarray:

    edges = z[...,[1,2,0],:] - z[...,[0,1,2],:]
    jacobian = edges[...,0,0] * edges[...,1,1] - edges[...,0,1] * edges[...,1,0]

    edge_lengths = np.linalg.norm(edges, axis=-1, ord=2)

    max_edge_length_product = np.max(
        edge_lengths[...,[0,1,2]] * edge_lengths[...,[1,2,0]]
    , axis=-1)

    scaled_jacobian = 2.0 / 3.0**0.5 * jacobian / max_edge_length_product

    return scaled_jacobian * orientation
