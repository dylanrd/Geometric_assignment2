import bpy
import inspect

from .deformation import *
from .extension import *

bl_info = {
    "name": "GDP Assignment 3 (Practical)",
    "author": "Add your names here!",
    "description": "A student implementation of Geometric Data Processing assignment 3 (practical).",
    "blender": (4, 1, 1),
    "version": (0, 0, 1),
    "location": "View3D",
    "warning": "",
    "category": "Mesh"
}

classes = [
    DifferentialCoordinateDeform,
    ConstrainedDifferentialCoordinateDeform,
    # TODO: For task 3, you should add your own Operators, Panels, or other UI elements here!
]


def register():
    for c in classes:
        try:
            bpy.utils.register_class(c)
        except AttributeError as e:
            print(
                f"Encountered an error while loading your {c.__name__} module, maybe you're missing some boilerplate?\n"
                f"\tError: '{e}'\n"
                f"\t(Take a look in '{inspect.getfile(c)}' to find out what's missing)"
            )

    deformation.register()
    extension.register()


def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
