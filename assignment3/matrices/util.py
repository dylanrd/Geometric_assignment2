import numpy as np

import bpy
import bmesh


def numpy_verts(mesh) -> np.ndarray:
    """
    Extracts a numpy array of (x, y, z) vertices from a blender mesh

    :param mesh: The BMesh to extract the vertices of.
    :return: A numpy array of shape [n, 3], where array[i, :] is the x, y, z coordinate of vertex i.
    """
    # Explained here:
    # https://blog.michelanders.nl/2016/02/copying-vertices-to-numpy-arrays-in_4.html
    if isinstance(mesh, bmesh.types.BMesh):
        data = bpy.data.meshes.new('tmp')
        mesh.to_mesh(data)
        mesh = data

    vertices = np.zeros(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', vertices)
    return vertices.reshape([len(mesh.vertices), 3])


def numpy_normals(mesh) -> np.ndarray:
    """
    Extracts a numpy array of (x, y, z) normals from a blender mesh

    :param mesh: The BMesh to extract the normals of.
    :return: A numpy array of shape [n, 3], where array[i, :] is the x, y, z normal of vertex i.
    """
    if isinstance(mesh, bmesh.types.BMesh):
        data = bpy.data.meshes.new('tmp')
        mesh.to_mesh(data)
        mesh = data

    normals = np.zeros(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('normal', normals)
    return normals.reshape([len(mesh.vertices), 3])


def set_verts(mesh, verts: np.ndarray):
    if isinstance(mesh, bmesh.types.BMesh):
        data = bpy.data.meshes.new('tmp1')
        mesh.to_mesh(data)
        data.vertices.foreach_set('co', verts.ravel())
        mesh.clear()
        mesh.from_mesh(data)
    else:
        mesh.vertices.foreach_set('co', verts.ravel())
