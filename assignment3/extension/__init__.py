import bpy
from mathutils import Vector
from .laplace_smoothing import *
from .hole_fill import *


class SelectVerticesNearCursorOperator(bpy.types.Operator):
    """Operator to select vertices near the cursor"""
    bl_idname = "object.select_vertices_near_cursor"
    bl_label = "Smooth vertices"
    bl_options = {'REGISTER', 'UNDO'}

    ## Adjust parameters
    tau: bpy.props.FloatProperty(
        name="ε", description="Minimum distance, below which the mesh is considered converged",
        min=-1.0, step=0.01, max=1.0, default=0.15
    )
    iterations: bpy.props.IntProperty(
        name="Iterations", description="Maximum number of iterations",
        min=1, max=200, default=5
    )

    @classmethod
    def poll(self, context):
        # Explicit Laplace Smoothing is only available when a mesh is selected
        return (
                context.view_layer.objects.active is not None
                and context.view_layer.objects.active.type == 'MESH'
        )

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):

        active_object = context.view_layer.objects.active

        # Produce BMesh types to work with
        mesh = bmesh.from_edit_mesh(active_object.data)

        # Gradient deformation expects a triangle mesh
        # self.status = f"Ensuring mesh contains only tris"
        bmesh.ops.triangulate(mesh, faces=mesh.faces)

        ##Selects the edges that are selected
        selected_edges_indices = [f.index for f in mesh.edges if f.select]
        # Selects the vertices that are selected
        selected_vertex_indices = [f.index for f in mesh.verts if f.select]

        # Smooth active mesh
        try:
            smoothed_mesh = iterative_explicit_laplace_smooth(
                mesh,
                self.tau,
                self.iterations, selected_vertex_indices,
                selected_edges_indices)

            verts = numpy_verts(mesh)

            # Substitute the smoothed selected edges into the vertices of the mesh
            for i, index in enumerate(selected_vertex_indices):

                verts[index] = smoothed_mesh[i]

            set_verts(mesh, verts)
            bmesh.update_edit_mesh(active_object.data)
        except Exception as error:
            self.report({'WARNING'}, f"Explicit Laplace Smoothing failed with error '{error}'")
            return {'CANCELLED'}

        # self.status = (f"Applied {self.iterations} iterations (ε={self.tau:.2f})")

        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout

        # Object selection
        row = layout.row(align=True)
        row.label(text="Object to smooth: ")
        row.separator()
        row.prop(context.view_layer.objects, 'active', text="", expand=True, emboss=False)
        layout.separator()

        # Convergence parameters
        layout.prop(self, 'iterations')
        layout.prop(self, 'tau')

        # layout.prop(self, 'status', text="Status", emboss=False)

    @staticmethod
    def menu_func(menu, context):
        menu.layout.operator(SelectVerticesNearCursorOperator.bl_idname)

class HoleFillOperator(bpy.types.Operator):
    """Operator to select vertices near the cursor"""
    bl_idname = "object.holefill_operator"
    bl_label = "fill hole"
    bl_options = {'REGISTER', 'UNDO'}

    tau: bpy.props.FloatProperty(
        name="ε", description="Minimum distance, below which the mesh is considered converged",
        min=0.0, step=0.01, max=1.0, default=0.15
    )
    iterations: bpy.props.IntProperty(
        name="Iterations", description="Maximum number of iterations",
        min=1, max=200, default=5
    )

    @classmethod
    def poll(self, context):
        # Explicit Laplace Smoothing is only available when a mesh is selected
        return (
                context.view_layer.objects.active is not None
                and context.view_layer.objects.active.type == 'MESH'
        )

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):

        active_object = context.view_layer.objects.active

        # Produce BMesh types to work with
        mesh = bmesh.from_edit_mesh(active_object.data)
        bmesh.ops.triangulate(mesh, faces=mesh.faces)

        # Smooth active mesh
        try:
            l00p = mesh_boundary_loops(mesh)
            ordered_loop = order_boundary_loop(mesh, l00p)

            for boundary_loop in ordered_loop:
                new_tris = hole_fill(boundary_loop)

                for edge in new_tris:
                    new_verts = [edge[0], edge[1], edge[2]]
                    if mesh.faces.get(new_verts) is None:
                        # print(f'new set {new_verts} added')
                        mesh.faces.new(new_verts)
                    # else:
                    #     print(f'set {new_verts} already exists')

            if bpy.context.mode == 'EDIT_MESH':
                bmesh.update_edit_mesh(active_object.data)
            else:
                mesh.to_mesh(active_object.data)

            active_object.data.update()

        except Exception as error:
            self.report({'WARNING'}, f"Explicit hole fill failed with error '{error}'")
            return {'CANCELLED'}

        # self.status = (f"Applied {self.iterations} iterations (ε={self.tau:.2f})")

        return {'FINISHED'}

    def draw(self, context):
        layout = self.layout

        # Object selection
        row = layout.row(align=True)
        row.label(text="Object to smooth: ")
        row.separator()
        row.prop(context.view_layer.objects, 'active', text="", expand=True, emboss=False)
        layout.separator()

        # Convergence parameters
        layout.prop(self, 'iterations')
        layout.prop(self, 'tau')

        # layout.prop(self, 'status', text="Status", emboss=False)

    @staticmethod
    def menu_func(menu, context):
        menu.layout.operator(HoleFillOperator.bl_idname)


def register():
    bpy.types.VIEW3D_MT_edit_mesh.append(SelectVerticesNearCursorOperator.menu_func)
    bpy.types.VIEW3D_MT_edit_mesh.append(HoleFillOperator.menu_func)


def unregister():
    bpy.utils.unregister_class(SelectVerticesNearCursorOperator)
    bpy.utils.unregister_class(HoleFillOperator)
