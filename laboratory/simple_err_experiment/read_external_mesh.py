from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def display_mesh(external_mesh_fol: Path):  # noqa: PLR0915
    coords_file = "node_coords"
    mesh_file = "mesh"
    abs_file = "absorbing_surface"
    free_surf_file = "free_surface"
    nc_adj_file = "nc_adjacencies"

    with (external_mesh_fol / coords_file).open() as f:
        n_nodes = int(f.readline().strip())
        coords = np.empty((n_nodes, 2))
        for i in range(n_nodes):
            pts = [float(k) for k in f.readline().split()]
            coords[i, :] = pts

    with (external_mesh_fol / mesh_file).open() as f:
        nelem = int(f.readline().strip())
        elem_nodes = np.empty((nelem, 9), dtype=np.int64)
        for i in range(nelem):
            elem_nodes[i, :] = [int(k) for k in f.readline().split()]
    elem_nodes -= 1

    with (external_mesh_fol / abs_file).open() as f:
        nabs = int(f.readline().strip())
        node_abs = np.empty((nabs, 2), dtype=np.int64)
        abs_elem = np.empty((nabs), dtype=np.int64)
        abs_edge = np.empty((nabs), dtype=np.uint8)
        for i in range(nabs):
            elem, nnode, node1, node2, edgetype = (
                int(k) for k in f.readline().split()
            )
            node_abs[i, :] = (node1 - 1, node2 - 1)
            abs_elem[i] = elem - 1
            abs_edge[i] = edgetype - 1

    with (external_mesh_fol / free_surf_file).open() as f:
        nafs = int(f.readline().strip())
        node_afs = np.empty((nafs, 2), dtype=np.int64)
        afs_elem = np.empty((nafs), dtype=np.int64)
        for i in range(nafs):
            elem, nnode, node1, node2 = (int(k) for k in f.readline().split())
            node_afs[i, :] = (node1 - 1, node2 - 1)
            afs_elem[i] = elem - 1

    with (external_mesh_fol / nc_adj_file).open() as f:
        nadj = int(f.readline().strip())
        elem_connect = np.empty((nadj, 2), dtype=np.int64)
        connect_type = np.empty((nadj,), dtype=np.uint8)
        edge_connect = np.empty((nadj,), dtype=np.uint8)
        for i in range(nadj):
            elem1, elem2, conntype, edge = (
                int(k) for k in f.readline().split()
            )
            elem_connect[i, :] = (elem1 - 1, elem2 - 1)
            edge_connect[i] = edge - 1
            connect_type[i] = conntype

    elem_node_coords = coords[elem_nodes, :]

    border_arr = [0, 4, 1, 5, 2, 6, 3, 7, 0]
    elem_border = elem_node_coords[:, border_arr, :]

    plt.plot(elem_border[..., 0].T, elem_border[..., 1].T, ":k")

    edge_to_inod = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
        ],
        dtype=np.uint8,
    )
    # abs_coords = coords[node_abs,:]
    node_abs2 = elem_nodes[abs_elem[:, None], edge_to_inod[abs_edge, :]]
    abs_coords = coords[node_abs2, :]
    plt.plot(abs_coords[..., 0].T, abs_coords[..., 1].T, "--r")

    afs_coords = coords[node_afs, :]
    plt.plot(afs_coords[..., 0].T, afs_coords[..., 1].T, "--b")

    connect_coords = coords[elem_nodes[elem_connect, 8], :]
    plt.plot(connect_coords[..., 0].T, connect_coords[..., 1].T, "--g")
