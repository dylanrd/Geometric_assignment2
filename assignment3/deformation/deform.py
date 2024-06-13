import scipy


import mathutils
from scipy.linalg import cho_solve, cho_factor

from assignment3.matrices.util import *
from assignment3.matrices.differential_coordinates import *


# !!! This function will be used for automatic grading, don't edit the signature !!!
def gradient_deform(mesh: bmesh.types.BMesh, A: mathutils.Matrix) -> np.ndarray:
    """
    Deforms a mesh by applying a transformation to its gradient vectors and then updating its vertices to match.

    This can be done with the help of the gradient, mass, and cotangent matrix functions from part 1.
    First find new gradients for the mesh by applying the transformation matrix A to its existing gradients,
    then use a sparse solver (from scipy) to find new vertices which match the target gradients.
    For more information, see the slides.

    :param mesh: The mesh to be modified.
    :param A: A 3x3 transformation matrix to apply to the gradients.
    :return: An Nx3 matrix representing new vertex positions for the mesh.
    """
    verts = numpy_verts(mesh)
    # TODO: Deform the gradients of the mesh and find new vertices.
    origin_barycenter = np.zeros(3)
    for vert in verts:
        origin_barycenter += vert
    origin_barycenter = origin_barycenter / len(verts)
    print(origin_barycenter)


    G = build_gradient_matrix(mesh)
    vert_x = verts[:, 0]
    vert_y = verts[:, 1]
    vert_z = verts[:, 2]

    G_x = G @ vert_x
    G_y = G @ vert_y
    G_z = G @ vert_z

    gradient = np.zeros((len(G_x), 3))
    gradient[:, 0] = G_x
    gradient[:, 1] = G_y
    gradient[:, 2] = G_z

    modified_gradients = gradient @ A

    # print("THIS IS GRADIENT", gradient)
    # print(gradient @ A)
    epsilon = 1e-12
    M, Mv = build_mass_matrices(mesh)
    A_solve = build_cotangent_matrix(G, Mv) + (epsilon * M)
    b_x = G.T @ Mv @ modified_gradients[:, 0]
    b_y = G.T @ Mv @ modified_gradients[:, 1]
    b_z = G.T @ Mv @ modified_gradients[:, 2]

    modified_vert_x = scipy.sparse.linalg.spsolve(A_solve, b_x)
    modified_vert_y = scipy.sparse.linalg.spsolve(A_solve, b_y)
    modified_vert_z = scipy.sparse.linalg.spsolve(A_solve, b_z)

    # c, low = cho_factor(np.array(A_solve))
    #
    # modified_vert_x = cho_solve((c, low), np.array(b_x))
    # modified_vert_y = cho_solve((c, low), np.array(b_y))
    # modified_vert_z = cho_solve((c, low), np.array(b_z))

    new_barycenter = np.zeros(3)
    for i in range(len(modified_vert_x)):
        new_barycenter += np.array([modified_vert_x[i], modified_vert_y[i], modified_vert_z[i]])
    new_barycenter = new_barycenter / len(modified_vert_x)

    translation_barycenter = origin_barycenter - new_barycenter
    print(translation_barycenter)

    res_verts = np.zeros((len(modified_vert_x), 3))
    res_verts[:, 0] = modified_vert_x + translation_barycenter[0]
    res_verts[:, 1] = modified_vert_y + translation_barycenter[1]
    res_verts[:, 2] = modified_vert_z + translation_barycenter[2]


    return res_verts


# !!! This function will be used for automatic grading, don't edit the signature !!!
def constrained_gradient_deform(
        mesh: bmesh.types.BMesh,
        selected_face_indices: list[int],
        A: mathutils.Matrix
) -> np.ndarray:
    """
    Deforms a mesh by applying a transformation to the gradient vectors of the selected triangles,
    and then updating its vertices to match.

    This can be done with the help of the gradient, mass, and cotangent matrix functions from part 1.
    First find new gradients by applying the transformation matrix to the gradients of only the selected triangles,
    then use a sparse solver (from scipy) to find new vertices which match the target gradients.
    For more information, see the slides.

    :param mesh: The mesh to be modified.
    :param selected_face_indices: List of indices indicating for which faces gradients should be changed.
    :param A: A 3x3 transformation matrix to apply to the gradients.
    :return: An Nx3 matrix representing new vertex positions for the mesh.
    """
    verts = numpy_verts(mesh)
    # TODO: Deform the gradients of the mesh and find new vertices.
    # origin_barycenter = np.zeros(3)
    # for vert in verts:
    #     origin_barycenter += vert
    # origin_barycenter = origin_barycenter / len(verts)
    # print(origin_barycenter)

    G = build_gradient_matrix(mesh)
    vert_x = verts[:, 0]
    vert_y = verts[:, 1]
    vert_z = verts[:, 2]

    G_x = G @ vert_x
    G_y = G @ vert_y
    G_z = G @ vert_z

    gradient = np.zeros((len(G_x), 3))
    gradient[:, 0] = G_x
    gradient[:, 1] = G_y
    gradient[:, 2] = G_z

    modified_gradients = gradient

    for selected_face_indice in selected_face_indices:
        triangle_gradient_vector_to_transform = modified_gradients[selected_face_indice*3:selected_face_indice*3+3,:]
        triangle_gradient_vector_transformed = triangle_gradient_vector_to_transform @ A
        modified_gradients[selected_face_indice*3:selected_face_indice*3+3,:] = triangle_gradient_vector_transformed

    # print("THIS IS GRADIENT", gradient)
    # print(gradient @ A)
    epsilon = 1e-12
    M, Mv = build_mass_matrices(mesh)
    A_solve = build_cotangent_matrix(G, Mv) + (epsilon * M)
    b_x = G.T @ Mv @ modified_gradients[:, 0]
    b_y = G.T @ Mv @ modified_gradients[:, 1]
    b_z = G.T @ Mv @ modified_gradients[:, 2]

    modified_vert_x = scipy.sparse.linalg.spsolve(A_solve, b_x)
    modified_vert_y = scipy.sparse.linalg.spsolve(A_solve, b_y)
    modified_vert_z = scipy.sparse.linalg.spsolve(A_solve, b_z)

    # c, low = cho_factor(np.array(A_solve))
    #
    # modified_vert_x = cho_solve((c, low), np.array(b_x))
    # modified_vert_y = cho_solve((c, low), np.array(b_y))
    # modified_vert_z = cho_solve((c, low), np.array(b_z))

    # new_barycenter = np.zeros(3)
    # for i in range(len(modified_vert_x)):
    #     new_barycenter += np.array([modified_vert_x[i], modified_vert_y[i], modified_vert_z[i]])
    # new_barycenter = new_barycenter / len(modified_vert_x)

    # translation_barycenter = origin_barycenter - new_barycenter
    # print(translation_barycenter)

    res_verts = np.zeros((len(modified_vert_x), 3))
    res_verts[:, 0] = modified_vert_x
    res_verts[:, 1] = modified_vert_y
    res_verts[:, 2] = modified_vert_z

    return res_verts
