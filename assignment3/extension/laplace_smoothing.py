import numpy
import numpy as np
import scipy.linalg
from scipy.sparse import coo_array, eye_array, sparray, diags

import bpy
import bmesh


def numpy_verts(mesh: bmesh.types.BMesh) -> np.ndarray:

    data = bpy.data.meshes.new('tmp')
    mesh.to_mesh(data)
    # Explained here:
    # https://blog.michelanders.nl/2016/02/copying-vertices-to-numpy-arrays-in_4.html
    vertices = np.zeros(len(mesh.verts) * 3, dtype=np.float64)
    data.vertices.foreach_get('co', vertices)
    return vertices.reshape([len(mesh.verts), 3])


def set_verts(mesh: bmesh.types.BMesh, verts: np.ndarray) -> bmesh.types.BMesh:
    data = bpy.data.meshes.new('tmp1')  # temp Blender Mesh to perform fast setting
    mesh.to_mesh(data)
    data.vertices.foreach_set('co', verts.ravel())
    mesh.clear()
    mesh.from_mesh(data)
    return mesh


def adjacency_matrix(mesh: bmesh.types.BMesh, selected_edges_indices: list[int], num_verts: int) -> coo_array:

    row = []
    col = []

    ## Take into account that there are only a few selected vertices of which its index will exceed the size
    max_vertex_index = max(max(edge.verts[0].index, edge.verts[1].index) for edge in mesh.edges)

    # Scale all vertex indices to fit within the [0, num_verts) range
    scaled_num_verts = min(num_verts, max_vertex_index)  # Ensure scaled_num_verts does not exceed num_verts
    scaling_factor = scaled_num_verts / max_vertex_index

    for edge in mesh.edges:
        for index in selected_edges_indices:
            if index == edge.index:
                # Scale the vertex indices to fit within the new range
                scaled_col = int(edge.verts[1].index * scaling_factor)
                scaled_row = int(edge.verts[0].index * scaling_factor)

                col.append(scaled_col)
                row.append(scaled_row)
                col.append(scaled_row)
                row.append(scaled_col)

    data = [1] * len(row)
    print(row, col)
    return coo_array((data, (row, col)), shape=(num_verts, num_verts))


# !!! This function will be used for automatic grading, don't edit the signature !!!
def build_combinatorial_laplacian(mesh: bmesh.types.BMesh, selected_edges_indices: list[int],
                                  num_verts: int) -> sparray:

    A = adjacency_matrix(mesh, selected_edges_indices, num_verts)

    # Convert the COO matrix to a dense matrix
    adjacency_mat = A.toarray()

    # Calculate the degree of each vertex
    degrees = np.sum(adjacency_mat, axis=1)

    # Create the degree matrix, use an epsilon as some diagonals might be 0
    epsilon = 1e-12
    degree_matrix = diags(degrees + epsilon)

    I = eye_array(num_verts)

    inv_d = diags(1 / degree_matrix.diagonal())

    res = I - (inv_d @ adjacency_mat)

    return res


# !!! This function will be used for automatic grading, don't edit the signature !!!
def explicit_laplace_smooth(
        vertices: np.ndarray,
        L: coo_array,
        tau: float,
) -> np.ndarray:

    res = vertices.copy()

    vertices2 = L @ vertices

    for i, vert in enumerate(vertices):
        res[i][0] = vert[0] - (tau * vertices2[i][0])
        res[i][1] = vert[1] - (tau * vertices2[i][1])
        res[i][2] = vert[2] - (tau * vertices2[i][2])
    return res


# !!! This function will be used for automatic grading, don't edit the signature !!!
def iterative_explicit_laplace_smooth(
        mesh: bmesh.types.BMesh,
        tau: float,
        iterations: int,
        selected_vertex_indices: list[int],
        selected_edge_indices: list[int]
) -> np.ndarray:

    # Compute combinatorial Laplace matrix
    L = build_combinatorial_laplacian(mesh, selected_edge_indices, len(selected_vertex_indices))

    #Select only the vertices that have been selected inside edit mode
    vertices = []
    for index in selected_vertex_indices:
        for verts in mesh.verts:
            if verts.index == index:
                vertices.append(verts.co)

    X = np.array(vertices)
    # Perform smoothing operations
    for _ in range(iterations):
        X = explicit_laplace_smooth(X, L, tau)

    return X
