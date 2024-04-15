import bpy
import shutil

# Ensure that the scene is empty
bpy.ops.wm.read_factory_settings(use_empty=True)

# Find blender binary
blender_bin = shutil.which("blender")
if blender_bin:
   print("Found:", blender_bin)
   bpy.app.binary_path = blender_bin
else:
   print("Unable to find blender!")

mesh = bpy.data.meshes.new(name="MyMesh")
