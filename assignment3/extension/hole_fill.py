import sys
from typing import List, Set

import numpy as np
from scipy.sparse import coo_array
from collections import deque

import bmesh


"""
    Computes the area in the selected triangle. Input needs to be an array of 3 BMVerts. 
"""
def calc_triangle_area(selected_tri) -> float:
    res = 0.5 * abs(np.linalg.norm(np.cross(selected_tri[1].co - selected_tri[0].co, selected_tri[2].co - selected_tri[0].co)))
    return res


""" 
    Triangulates the weights from the one boundary loop we have here
"""
def weight_triangulation(boundary_loop: list[bmesh.types.BMVert]):
    n = len(boundary_loop)

    # We are using area as the weight
    weights = [np.zeros(i) for i in range(n - 1, 0, -1)]
    weights[1] = np.array([calc_triangle_area(boundary_loop[i:i+3]) for i in range(n - 2)])
    lambdas = [np.zeros(i if i < n - 2 else 0, dtype=int) for i in range(n - 1, 0, -1)]  # Indexing offset.

    # idk why the paper says to start from j = 2, then immediately do j = j + 1, we just start from 3
    for j in range(3, n):
        for i in range(n - j):
            k = i + j
            min_weight = sys.float_info.max
            optimal_m = None
            for m in range(j - 1):
                # We made the boundary triangular to save a bit of space so we need to re assign the locations correctly
                shifted_m = j - m - 2
                shifted_i = m + i + 1

                Weight_ik = weights[m][i] + weights[shifted_m][shifted_i] + calc_triangle_area([boundary_loop[i], boundary_loop[shifted_i], boundary_loop[k]])

                if Weight_ik < min_weight:
                    min_weight = Weight_ik
                    optimal_m = m

            weights[j - 1][i] = min_weight
            lambdas[j - 1][i] = i + 1 + optimal_m

    return lambdas


"""
    Fills the hole which is encased in the given boundary loop. The boundary loop requires you to have all vertices in order.
"""
def hole_fill(boundary_loop: list[bmesh.types.BMVert]):
    lambdas = weight_triangulation(boundary_loop)

    # Reconstruct triangulation.
    sections = deque([(0, len(boundary_loop) - 1)])
    triangles = []

    while sections:
        d, b = sections.pop()
        if b - d == 2:
            optimal_m = d + 1
        else:
            optimal_m = lambdas[b - d - 1][d]
        triangles.append((d, optimal_m, b))
        if 1 < optimal_m - d:
            sections.append((d, optimal_m))
        if 1 < b - optimal_m:
            sections.append((optimal_m, b))

    res = []
    for tri in triangles:
        res.append([boundary_loop[tri[0]], boundary_loop[tri[1]], boundary_loop[tri[2]]])
    return res


"""
    Takes a set of unordered boundary loops and returns ordered ones, where the vertex indices are connected in a loop. 
    This is definitely not an efficient way to do this but whatever tbh
    mesh - The mesh of our object
    boundary_loops - the boundary loop set which were created from the mesh_boundary_loops function.
"""
def order_boundary_loop(mesh:bmesh.types.BMesh, boundary_loops: List[Set[bmesh.types.BMEdge]]) -> List[List[bmesh.types.BMVert]]:
    res = []

    # Go through all the found boundary loops
    for loop in boundary_loops:
        loop_list = list(loop)
        current_elem = loop_list[0]
        ordered_list = []
        last_added_vertex = None

        # While loop which lets us go through the mesh in any order we want.
        # We stop when loop list still has one element in it because the first edge adds two vertices instead of just one.
        while len(loop_list) > 1:
            # Just a precaution against infinite loops, shouldn't be a problem but still nice to have.
            gud = False

            for edge in linked_edges(mesh, current_elem):

                # If the edge is connected to the current one, we add it to the list and remove it from the loop list so we know it has already been traversed
                if edge in loop_list:
                    loop_list.remove(current_elem)

                    if current_elem.verts[0] == edge.verts[0] or current_elem.verts[0] == edge.verts[1]:
                        # This if check is basically just to put the first non connected vertex in as well.
                        # If you want to move it to when len(loop_list) is 1 that's also fine but who cares really
                        if current_elem.verts[1] != last_added_vertex:
                            ordered_list.append(current_elem.verts[1])
                        ordered_list.append(current_elem.verts[0])
                        last_added_vertex = current_elem.verts[0]

                    else:
                        if current_elem.verts[0] != last_added_vertex:
                            ordered_list.append(current_elem.verts[0])
                        ordered_list.append(current_elem.verts[1])
                        last_added_vertex = current_elem.verts[1]

                    current_elem = edge
                    gud = True
                    break
            if not gud:
                raise Exception("infinite loop made")
        res.append(ordered_list)
    return res


# This was just taken from our assignment 1
def mesh_boundary_loops(mesh: bmesh.types.BMesh) -> List[Set[bmesh.types.BMEdge]]:
    """
    Finds the boundary loops of a BMesh.

    Each boundary loop is represented by a `set()` of the edges which make it up.
    Each edge appears in at most one boundary loop.
    Non-boundary edges will not appear in any boundary loops
    HINT: how do you check if an edge is on a boundary?

    :param mesh: The mesh to find the boundary loops of.
    :return: A list of boundary loops, each of which is a set of `BMEdge`s.
    """
    # TODO: Find the boundary loops of the mesh
    boundary_loops = []
    visited_edges = []
    queue = []

    for edge in mesh.edges:

        if boundary_checker(mesh, edge) and edge not in visited_edges:
            loop = set()
            loop.add(edge)
            visited_edges.append(edge)
            queue.append(edge)
            while len(queue) > 0:
                edge = queue.pop(0)
                for connected_edges in linked_edges(mesh, edge):

                    if boundary_checker(mesh, connected_edges) and connected_edges not in visited_edges:
                        loop.add(connected_edges)
                        visited_edges.append(connected_edges)
                        queue.append(connected_edges)

            boundary_loops.append(loop)

    return boundary_loops


def linked_edges(mesh: bmesh.types.BMesh, find_edge) -> List[bmesh.types.BMEdge]:
    connected_edges = []

    for edge in mesh.edges:
        if find_edge == edge:
            continue
        if find_edge.verts[0] == edge.verts[1] or find_edge.verts[0] == edge.verts[0]:
            connected_edges.append(edge)

        if find_edge.verts[1] == edge.verts[1] or find_edge.verts[1] == edge.verts[0]:
            connected_edges.append(edge)

    return connected_edges


def boundary_checker(mesh: bmesh.types.BMesh, find_edge) -> bool:
    count = 0
    for face in mesh.faces:

        if find_edge.verts[0] in face.verts and find_edge.verts[1] in face.verts:
            count += 1

    if count == 1:
        return True
    else:
        return False

