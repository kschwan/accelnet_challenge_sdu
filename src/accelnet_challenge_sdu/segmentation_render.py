import cv2 as cv
import numpy as np
import open3d as o3d
import os
import rospkg

from .pose import Pose


class GroundTruthSegmentationModel:
    def __init__(self, camera_model):

        self.camera_model = camera_model

        package_path = rospkg.RosPack().get_path('accelnet_challenge_sdu')

        # Open3D didn't like the original Needle.OBJ, so I re-exported from Blender with the "Triangulated Mesh" option set
        meshes = {
            'needle': o3d.io.read_triangle_mesh(os.path.join(package_path, 'resources', 'meshes', 'needle.obj'), enable_post_processing=True),
            'toolrolllink': o3d.io.read_triangle_mesh(os.path.join(package_path, 'resources', 'meshes', 'tool roll link.STL'), enable_post_processing=True),
            'toolpitchlink': o3d.io.read_triangle_mesh(os.path.join(package_path, 'resources', 'meshes', 'tool pitch link.STL'), enable_post_processing=True),
            'toolgripper1link': o3d.io.read_triangle_mesh(os.path.join(package_path, 'resources', 'meshes', 'tool gripper1 link.STL'), enable_post_processing=True),
            'toolgripper2link': o3d.io.read_triangle_mesh(os.path.join(package_path, 'resources', 'meshes', 'tool gripper2 link.STL'), enable_post_processing=True),
            'thread segment': o3d.geometry.TriangleMesh.create_cylinder(radius=0.00372/2, height=0.0275),  # Values from Blender model
        }

        # Paint meshes in distinct colors
        meshes['needle'].paint_uniform_color((1, 0, 0))           # needle -> red
        meshes['toolrolllink'].paint_uniform_color((0, 0, 1))     # tool shaft -> blue
        meshes['toolpitchlink'].paint_uniform_color((0, 1, 0))    # tool wrist -> green
        meshes['toolgripper1link'].paint_uniform_color((0, 1, 0)) # tool gripper -> green
        meshes['toolgripper2link'].paint_uniform_color((0, 1, 0)) # tool gripper -> green
        meshes['thread segment'].paint_uniform_color((1, 1, 0))   # thread -> yellow

        # We render a 3D scene that mirrors the one in AMBF
        self.render = o3d.visualization.rendering.OffscreenRenderer(*self.camera_model.size)
        # render.scene.show_axes(True)
        self.render.scene.set_background([0, 0, 0, 0])
        self.render.scene.set_lighting(self.render.scene.LightingProfile.NO_SHADOWS, [0, 0, 0])

        # Keys match AMBF identifiers
        self.geometries = {
            'Needle': meshes['needle'],
            'psm1/toolrolllink': meshes['toolrolllink'],
            'psm1/toolpitchlink': meshes['toolpitchlink'],
            'psm1/toolgripper1link': meshes['toolgripper1link'],
            'psm1/toolgripper2link': meshes['toolgripper2link'],
            'psm2/toolrolllink': meshes['toolrolllink'],
            'psm2/toolpitchlink': meshes['toolpitchlink'],
            'psm2/toolgripper1link': meshes['toolgripper1link'],
            'psm2/toolgripper2link': meshes['toolgripper2link'],
        }

        # Thread segments
        self.geometries['Thread'] = meshes['thread segment']

        for i in range(1, 40):
            self.geometries[f'Thread_{i:03d}'] = meshes['thread segment']

        for name, mesh in self.geometries.items():
            self.render.scene.add_geometry(name, mesh, o3d.visualization.rendering.MaterialRecord())


    def generate_renderings(self, tf_buffer, stamp):
        # Move geometries according to current state of AMBF
        for name in self.geometries:
            tf_msg = tf_buffer.lookup_transform('world', name, stamp)
            tf = Pose.from_msg(tf_msg.transform).to_matrix()
            self.render.scene.set_geometry_transform(name, tf)

        # View from left cam
        tf_msg = tf_buffer.lookup_transform('cameraL_cv', 'world', stamp)
        T_left_world = Pose.from_msg(tf_msg.transform).to_matrix()
        self.render.setup_camera(self.camera_model.intrinsic_matrix, T_left_world, *self.camera_model.size)
        im_render_l = np.asarray(self.render.render_to_image())

        # View from right cam
        tf_msg = tf_buffer.lookup_transform('cameraR_cv', 'world', stamp)
        T_right_world = Pose.from_msg(tf_msg.transform).to_matrix()
        self.render.setup_camera(self.camera_model.intrinsic_matrix, T_right_world, *self.camera_model.size)
        im_render_r = np.asarray(self.render.render_to_image())

        return im_render_l, im_render_r

    def postprocess_rendering(self, im_render):
        conversions = [
        #   lower (RGB)     upper (RBG)     new color (BGR)
            ((0, 0, 150),   (80, 80, 255),  (255, 0, 0)),   # tool shaft (blue)
            ((0, 180, 0),   (150, 255, 80), (0, 255, 0)),   # tool wrist/gripper (green)
            ((150, 150, 0), (255, 255, 80), (0, 255, 255)), # thread (yellow)
            ((50, 0, 0),    (255, 20, 20),  (0, 0, 255)),   # needle (red)
        ]

        im_segmask = np.zeros(im_render.shape, dtype=np.uint8)

        for lower, upper, color in conversions:
            mask = cv.inRange(im_render, lower, upper)
            im_segmask[np.nonzero(mask)] = color

        return im_segmask

    def generate_masks(self, tf_buffer, stamp):
        im_render_l, im_render_r = self.generate_renderings(tf_buffer, stamp)
        return self.postprocess_rendering(im_render_l), self.postprocess_rendering(im_render_r)
