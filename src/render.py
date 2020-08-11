import numpy as np
import time
import os
import math
import pyglet
import cv2
pyglet.options['shadow_window'] = False
import trimesh
from threading import Thread, Lock
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer


def rotationx(theta):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0],
        [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def rotationy(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])


def location(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ])


if __name__ == "__main__":

    scene = Scene(ambient_light=np.array([0.1, 0.1, 0.1, 1.0]))

    cam = PerspectiveCamera(yfov=(np.pi / 3.0))
    cam_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    points_mesh = Mesh.from_points(np.array([[0.0, 0.0, 0.0]]))
    human_node = scene.add(points_mesh)
    plane_size = 0.7
    plane_points = np.array([
        [plane_size, -0.65, plane_size],
        [plane_size, -0.65, -plane_size],
        [-plane_size, -0.65, plane_size],
        [-plane_size, -0.65, -plane_size]
    ])
    plane_faces = np.array([
        [0, 1, 2],
        [2, 1, 3]
    ])
    plane_colors = np.ones((4, 3), dtype=np.float32) / 3.0
    plane_mesh = trimesh.Trimesh(vertices=plane_points, faces=plane_faces, vertex_colors=plane_colors)
    plane_mesh = Mesh.from_trimesh(plane_mesh, smooth=False)
    plane_node = scene.add(plane_mesh)
    direc_l = DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node1 = scene.add(direc_l, pose=np.matmul(rotationx(20), rotationy(60)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node2 = scene.add(direc_l, pose=np.matmul(rotationx(-20), rotationy(180)))
    direc_l = DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node3 = scene.add(direc_l, pose=np.matmul(rotationx(20), rotationy(-60)))
    point_l = PointLight(color=np.ones(3), intensity=10.0)
    light_node4 = scene.add(point_l, pose=location(1.0, 1.5, 1.7))
    point_l = PointLight(color=np.ones(3), intensity=10.0)
    light_node4 = scene.add(point_l, pose=location(-1.0, 1.5, 1.7))
    point_l = PointLight(color=np.ones(3), intensity=10.0)
    light_node4 = scene.add(point_l, pose=location(0.0, 1.5, -2.0))
    cam_node = scene.add(cam, pose=cam_pose)
    render_flags = {
        'flip_wireframe': False,
        'all_wireframe': False,
        'all_solid': False,
        'shadows': True,
        'vertex_normals': False,
        'face_normals': False,
        'cull_faces': True,
        'point_size': 1.0,
    }
    viewer_flags = {
        'mouse_pressed': False,
        'rotate': False,
        'rotate_rate': np.pi / 6.0,
        'rotate_axis': np.array([0.0, 1.0, 0.0]),
        'view_center': np.array([0.0, 0.0, 0.0]),
        'record': False,
        'use_raymond_lighting': False,
        'use_direct_lighting': False,
        'lighting_intensity': 3.0,
        'use_perspective_cam': True,
        'save_directory': '/home/wanglz14/Desktop/video/',
        'window_title': 'FOV Human',
        'refresh_rate': 10.0,
        'fullscreen': False,
        'show_world_axis': False,
        'show_mesh_axes': False,
        'caption': None
    }
    v = Viewer(scene, viewport_size=(800, 800), render_flags=render_flags, viewer_flags=viewer_flags, run_in_thread=True)
    mesh = trimesh.load('../results/test/01.ply')
    points_mesh = Mesh.from_trimesh(mesh, smooth=True)
    v._render_lock.acquire()
    scene.remove_node(human_node)
    human_node = scene.add(points_mesh)
    v._render_lock.release()
