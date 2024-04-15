import bpy
from dataclasses import dataclass, field
import numpy as np
import bmesh

def __extract_kwargs_from_dataclass(data_class) -> dict:
    kwargs = {key: value for key, value in data_class.__dict__.items()
        if not key.startswith('__')
        and not callable(key)
    }
    return kwargs

@dataclass
class SceneSettings:
    pixels_x: int = 1200
    pixels_y: int = 1200
    filepath = "./thisImage.png"

def create_empty_scene(settings: SceneSettings):
    # Remove all initial objects
    cube = bpy.data.objects["Cube"]
    bpy.data.objects.remove(cube)

    bpy.context.scene.render.filepath = settings.filepath
    bpy.context.scene.render.resolution_x = settings.pixels_x #perhaps set resolution in code
    bpy.context.scene.render.resolution_y = settings.pixels_y

@dataclass
class CameraSettings:
    # See https://docs.blender.org/api/current/bpy.ops.object.html#bpy.ops.object.camera_add
    location: list[float] = field(default_factory=lambda: [0, 0, 0])
    rotation: list[float] = field(default_factory=lambda: [0, 0, 0])
    scale: list[float] = field(default_factory=lambda: [0, 0, 0])

def modify_camera(camera_settings: CameraSettings):
    kwargs = __extract_kwargs_from_dataclass(camera_settings)
    # bpy.ops.object.camera_add(**kwargs)

@dataclass
class SphereSettings:
    name: str = "BasicSphere"
    location: list[float] = field(default_factory=lambda:[0,0,0])
    radius: float = 0.5
    subdivisions: int = 3
    grainer: float = 0.15

def insert_sphere(sphere_settings: SphereSettings):
    kwargs = __extract_kwargs_from_dataclass(sphere_settings)
    # bpy.ops.mesh.primitive_ico_sphere_add(**kwargs)
    
    # Create new mesh and object
    mesh = bpy.data.meshes.new(sphere_settings.name)
    basic_sphere = bpy.data.objects.new(sphere_settings.name + "_mesh", mesh)

    # Add object to the scene
    bpy.context.collection.objects.link(basic_sphere)

    # Select newly created object
    bpy.context.view_layer.objects.active = basic_sphere
    basic_sphere.select_set(True)

    # construct bmesh sphere and assign it to blender mesh
    bm = bmesh.new()
    bmesh.ops.create_icosphere(
        bm,
        subdivisions=sphere_settings.subdivisions,
        radius=sphere_settings.radius,
    )

    # modify vertices
    for v in bm.verts:
        # Randomize size
        r = (1.0 - sphere_settings.grainer + 2.0 * sphere_settings.grainer * np.random.rand(3))
        v.co.x *= r[0] 
        v.co.y *= r[1] 
        v.co.z *= r[2] 

        # Shift position
        v.co.x += sphere_settings.location[0]
        v.co.y += sphere_settings.location[1]
        v.co.z += sphere_settings.location[2]

    mat = bpy.data.materials.new('MyMaterial')
    mat.diffuse_color = (237/255, 135/255, 208/255, 1.0)
    basic_sphere.active_material = mat

    bm.to_mesh(mesh)
    bm.free()

    # Some more settings
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.ops.object.shade_smooth()

def render_all():
    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    # Create empty scene
    scene_settings = SceneSettings()
    create_empty_scene(scene_settings)

    # Define camera
    camera_settings = CameraSettings()
    modify_camera(camera_settings)

    # positions = np.linspace(-1, 1, 30).reshape((-1, 3))
    positions = -2.0 + 4.0*np.random.rand(30, 3)
    for i in range(positions.shape[0]):
        location = positions[i]
        sphere_settings = SphereSettings(
            location=location,
        )
        insert_sphere(sphere_settings)

    render_all()

