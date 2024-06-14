import numpy
import numpy as np
import scipy.linalg
from scipy.sparse import coo_array, eye_array, sparray, diags

import bpy
import bmesh


def numpy_verts(mesh: bmesh.types.BMesh) -> np.ndarray:
    """
    Extracts a numpy array of (x, y, z) vertices from a blender mesh

    :param mesh: The BMesh to extract the vertices of.
    :return: A numpy array of shape [n, 3], where array[i, :] is the x, y, z coordinate of vertex i.
    """
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


# HINT: This is a helper method which you can change (for example, if you want to try different sparse formats)
def adjacency_matrix(mesh: bmesh.types.BMesh, selected_edges_indices: list[int], num_verts: int) -> coo_array:
    # HINT: Iterating over mesh.edges is significantly faster than iterating over mesh.verts and getting neighbors!
    #       Building a sparse matrix from a set of I, J, V triplets is also faster than adding elements sequentially.
    # TODO: Create a sparse adjacency matrix using one of the types from scipy.sparse

    row = []
    col = []
    # for edge in mesh.edges:
    #     for index in selected_edges_indices:
    #         if index == edge.index:
    #             col.append(edge.verts[1].index)
    #             row.append(edge.verts[0].index)
    #             col.append(edge.verts[0].index)
    #             row.append(edge.verts[1].index)
    #
    # data = [1] * len(row)
    # print(selected_edges_indices)
    # print(data, row, col)

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
    # TODO: Build the combinatorial laplacian matrix

    A = adjacency_matrix(mesh, selected_edges_indices, num_verts)

    # Convert the COO matrix to a dense matrix
    adjacency_mat = A.toarray()
    print(A)



    # Calculate the degree of each vertex
    degrees = np.sum(adjacency_mat, axis=1)

    # Create the degree matrix
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
    """
    Performs smoothing of a list of vertices given a combinatorial Laplace matrix and a weight Tau.

    Updates are computed using the laplacian matrix and then weighted by Tau before subtracting from the vertices.

        x = x - tau * L @ x

    :param vertices: Vertices to apply offsets to as an Nx3 numpy array.
    :param L: The NxN sparse laplacian matrix
    :param tau: Update weight, tau=0 leaves the vertices unchanged, and tau=1 applies the full update.
    :return: The new positions of the vertices as an Nx3 numpy array.
    """
    # TODO: Update the vertices using the combinatorial laplacian matrix L
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
    """
    Performs smoothing of a given mesh using the iterative explicit Laplace smoothing.

    First, we define the coordinate vectors and the combinatorial Laplace matrix as numpy arrays.
    Then, we apply the smoothing operation as many times as iterations.
    We weight the updating vector in each iteration by tau.

    :param selected_vertex_indices:
    :param mesh: Mesh to smooth.
    :param tau: Update weight.
    :param iterations: Number of smoothing iterations to perform.
    :return: A mesh with the updated coordinates after smoothing.
    """

    # Get coordinate vectors as numpy arrays
    # X = numpy_verts(mesh)

    # Compute combinatorial Laplace matrix
    L = build_combinatorial_laplacian(mesh, selected_edge_indices, len(selected_vertex_indices))

    vertices = []
    for index in selected_vertex_indices:
        for verts in mesh.verts:
            if verts.index == index:
                vertices.append(verts.co)

    X = np.array(vertices)
    # Perform smoothing operations
    for _ in range(iterations):
        X = explicit_laplace_smooth(X, L, tau)

    # Write smoothed vertices back to output mesh
    # set_verts(mesh, X)
    print("AAAAAAAAAAAAAAAAA")
    return X
