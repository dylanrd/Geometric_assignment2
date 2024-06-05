import mathutils

from .deform import *
from assignment3.matrices.test import *


# todo: this needs a better name!
class DifferentialCoordinateDeformBase(bpy.types.Operator):
    bl_options = {'REGISTER', 'UNDO'}

    # Input parameters
    matrix_selection_mode: bpy.props.EnumProperty(
        name="Matrix Selection Mode", description="Method for setting up the gradient transformation matrix A.",
        items=[
            ('CUSTOM', "Custom Matrix", ""),
            ('ROTATION', "Rotation", ""),
            ('SCALE', "Scale (X, Y, Z)", ""),
        ]
    )
    A_matrix: bpy.props.FloatVectorProperty(
        name="A (matrix)",
        description="Gradient Transformation Matrix",
        size=[3, 3],
        subtype='MATRIX',
        default=mathutils.Matrix.Identity(3)
    )
    A_rotation: bpy.props.FloatVectorProperty(
        name="A (rotation)",
        description="Gradient Transformation by Rotation",
        size=[3],
        subtype='DIRECTION',
        default=[0, 0, 1]
    )
    A_scale: bpy.props.FloatVectorProperty(
        name="A (scale)",
        description="Gradient Transformation by Scale",
        size=[3],
        subtype='XYZ_LENGTH',
        default=[1, 1, 1]
    )

    # Output parameters
    status: bpy.props.StringProperty(
        name="Smoothing Status", default="Status not set"
    )

    def A(self):
        if self.matrix_selection_mode == 'ROTATION':
            self.A_matrix = mathutils.Vector(self.A_rotation).rotation_difference([0, 0, 1]).to_matrix()
        if self.matrix_selection_mode == 'SCALE':
            self.A_matrix = mathutils.Matrix([
                [self.A_scale.x, 0, 0],
                [0, self.A_scale.y, 0],
                [0, 0, self.A_scale.z],
            ])
        return self.A_matrix

    @classmethod
    def poll(cls, context):
        return (
                context.view_layer.objects.active is not None
                and context.view_layer.objects.active.type == 'MESH'
        )

    def draw(self, context):
        layout = self.layout

        row = layout.row(align=True)
        row.label(text="Object to deform: ")
        row.separator()
        row.prop(context.view_layer.objects, 'active', text="", expand=True, emboss=False)
        layout.separator()

        layout.prop(self, 'matrix_selection_mode')
        if self.matrix_selection_mode == 'CUSTOM':
            layout.prop(self, 'A_matrix', text='')
        elif self.matrix_selection_mode == 'ROTATION':
            layout.prop(self, 'A_rotation', text='')
        elif self.matrix_selection_mode == 'SCALE':
            layout.prop(self, 'A_scale', text='')

        layout.prop(self, 'status', text="Status", emboss=False)


class DifferentialCoordinateDeform(DifferentialCoordinateDeformBase):
    bl_idname = "object.differential_deform"
    bl_label = "Mesh Gradient Deformation"

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        active_object = context.view_layer.objects.active

        # Produce BMesh types to work with
        mesh = bmesh.new()
        mesh.from_mesh(active_object.data)

        # Gradient deformation expects a triangle mesh
        self.status = f"Ensuring mesh contains only tris"
        bmesh.ops.triangulate(mesh, faces=mesh.faces)

        # Apply the deformation
        self.status = f"Computing deformation"
        new_verts = gradient_deform(mesh, self.A())
        set_verts(mesh, new_verts)

        # Write the results back to the underlying mesh
        self.status = f"Updating Mesh"
        mesh.to_mesh(active_object.data)
        active_object.data.update()

        self.status = f"Done"
        return {'FINISHED'}

    @staticmethod
    def menu_func(menu, context):
        menu.layout.operator(DifferentialCoordinateDeform.bl_idname)


class ConstrainedDifferentialCoordinateDeform(DifferentialCoordinateDeformBase):
    bl_idname = "object.constrained_differential_deform"
    bl_label = "Constrained Mesh Gradient Deformation"

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        active_object = context.view_layer.objects.active

        # Produce BMesh types to work with
        mesh = bmesh.from_edit_mesh(active_object.data)

        # Gradient deformation expects a triangle mesh
        self.status = f"Ensuring mesh contains only tris"
        bmesh.ops.triangulate(mesh, faces=mesh.faces)

        # Determine selected faces
        selected_face_indices = [f.index for f in mesh.faces if f.select]
        self.num_selected_faces = len(selected_face_indices)

        # Apply the deformation
        new_verts = constrained_gradient_deform(mesh, selected_face_indices, self.A())

        # Update the original mesh
        self.status = f"Updating Mesh"
        set_verts(mesh, new_verts)
        bmesh.update_edit_mesh(active_object.data)

        self.status = f"Done"
        return {'FINISHED'}

    @staticmethod
    def menu_func(menu, context):
        menu.layout.operator(ConstrainedDifferentialCoordinateDeform.bl_idname)


def register():
    bpy.types.VIEW3D_MT_object.append(DifferentialCoordinateDeform.menu_func)
    bpy.types.VIEW3D_MT_edit_mesh.append(ConstrainedDifferentialCoordinateDeform.menu_func)
