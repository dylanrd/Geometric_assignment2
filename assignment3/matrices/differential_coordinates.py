import numpy
import numpy as np
from scipy.sparse import coo_array, eye_array, sparray

import bpy
import bmesh


# !!! This function will be used for automatic grading, don't edit the signature !!!
def triangle_gradient(triangle: bmesh.types.BMFace) -> np.ndarray:
    """
    Computes the local gradient of a triangular face.

    The local gradient $g$ is a 3x3 matrix where each column is the cross product of the triangle's normal
    with the vector representing one of its three edges, divided by half of its area, so:

        g = [(N \cross e_1) (N \cross e_2) (N \cross e_3)] / 2A

    Where $N$ is the triangle's normal, $e_i$ is the ith edge of the triangle, and A is the triangle's area.

    :param triangle: Triangular face to find the local gradient of.
    :return: A 3x3 gradient matrix.
    """
    assert len(triangle.verts) == 3
    normal = np.array(triangle.normal)
    local_gradient = numpy.zeros([3, 3])
    # TODO: Find the local gradient for this triangle.
    return local_gradient


# !!! This function will be used for automatic grading, don't edit the signature !!!
def build_gradient_matrix(mesh: bmesh.types.BMesh) -> sparray:
    """
    Computes the gradient matrix $G$ for a triangular mesh.

    The local gradient of each triangle in the mesh appears in the overall mesh gradient matrix,
    but its rows are distributed along the columns according to the index of the vertex they are associated with.
    For more information, see the slides.

    :param mesh: Triangular mesh to find the gradient matrix of.
    :return: A 3MxN gradient matrix,
             where M and N are the number of triangles and number of vertices in the mesh, respectively.
    """
    num_faces, num_verts = len(mesh.faces), len(mesh.verts)
    # TODO: construct the sparse gradient matrix for the mesh
    return coo_array(([], ([], [])), shape=(num_faces * 3, num_verts))


# !!! This function will be used for automatic grading, don't edit the signature !!!
def build_mass_matrices(mesh: bmesh.types.BMesh) -> tuple[sparray, sparray]:
    """
    Computes the mass matrices $M$ and $Mv$ for a triangular mesh.

    In both mass matrices, elements only appear along the diagonal, all other elements are zero:

        $M_ii$ is the sum of the area of each triangle connected to vertex $i$,

        $Mv_(3i+l)(3i+l)$ is the area of triangle $i$, where $l$ is the index (0, 1, 2) of each vertex of the triangle.

    For more information, see the slides.

    :param mesh: Triangular mesh to find the mass matrices of.
    :return: A tuple containing the NxN sparse matrix $M$ and the 3Mx3M sparse matrix $Mv$,
             where M and N are the number of triangles and number of vertices in the mesh, respectively.
    """
    num_faces, num_verts = len(mesh.faces), len(mesh.verts)
    # TODO: construct the mass matrices M and Mv for the mesh
    return (
        coo_array(([], ([], [])), shape=(num_verts, num_verts)),
        coo_array(([], ([], [])), shape=(3 * num_faces, 3 * num_faces))
    )


# !!! This function will be used for automatic grading, don't edit the signature !!!
def build_cotangent_matrix(G: sparray, Mv: sparray) -> sparray:
    """
    Computes the cotangent matrix $S$ from the gradient and mass matrices.

    :param G: A 3MxN gradient matrix.
    :param Mv: A 3Mx3M mass matrix.
    :return: A 3Mx3M cotangent matrix.
    """
    # TODO: find the cotangent matrix S based on G and Mv
    return eye_array(num_verts)
