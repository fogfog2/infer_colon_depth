# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import numpy as np
import os

import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as funct
import cv2
from glob import glob
from cv2 import imwrite, applyColorMap
import PIL.Image as pil

from packnet_sfm.models.model_wrapper_custom import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.horovod import hvd_init, rank, world_size, print0
from packnet_sfm.utils.image import load_image, image_grid
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.logging import pcolor
from packnet_sfm.networks.layers.resnet.layers import disp_to_depth
from packnet_sfm.geometry.pose import Pose
from packnet_sfm.geometry.camera import Camera

import threading
import time

i= 0
pcd_list = []
view_pcd_list = []
axis_set_list = []
trajectory_list = []
fov_set_list = []
timer2 = 0

prev_axis = []

SHOW_AXIS = True
SHOW_TRAJECTORY = False
SHOW_FOV = False
SHOW_PCD = False
SHOW_VIEW_PCD = True
SHOW_NAVI_TRIANGLE = False
USE_POSE_PREDICT = False

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2


class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle

                rotaxis = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a)
                cylinder_segment = cylinder_segment.rotate(
                    R=rotaxis)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def update_line_mesh(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius        
        self.cylinder_segments = []


        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                rotaxis = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a)
                cylinder_segment = cylinder_segment.rotate(
                    R=rotaxis)
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)
            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)

    def update_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.update_geometry(cylinder)



def is_image(file, ext=('.png', '.jpg',)):
    """Check if a file is an image with certain extensions"""
    return file.endswith(ext)


def parse_args():
    parser = argparse.ArgumentParser(description='PackNet-SfM inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)')
    parser.add_argument('--input', type=str, help='Input file or folder')
    parser.add_argument('--output', type=str, help='Output file or folder')
    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')
    parser.add_argument('--half', action="store_true", help='Use half precision (fp16)')
    parser.add_argument('--save', type=str, choices=['npz', 'png'], default=None,
                        help='Save format (npz or png). Default is None (no depth map is saved).')
    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    assert (is_image(args.input) ) or \
           (not is_image(args.input) and not is_image(args.input)), \
        'Input and output must both be images or folders'
    return args

def path_loader(pts):

    min_z = pts[:,2].min()
    max_z = pts[:,2].max()
    step = 4
    res = (max_z - min_z)/step

    path = [[0,0,0]]
    for i in range(0,step):
        step_in = min_z + res*(i) 
        step_out = min_z + res*(i+1)
        idx = np.where(pts[:,2]< step_out)
        subptx = pts[idx]
        idx = np.where(subptx[:,2]>=step_in)
        test = subptx[idx]
        x = test[:,0].mean()
        y = test[:,1].mean()
        z = (step_in + step_out)/2.0
        path.append([x,y,z])
    return path

def path_loader_inv(pts):

    min_z = pts[:,2].min()
    max_z = pts[:,2].max()
    step = 10
    res = (max_z - min_z)/step

    path = [[0,0,0]]
    #path = []
    for i in range(9,step):
        step_in = max_z - res*(i+1) 
        step_out = max_z - res*(i)
        idx = np.where(pts[:,2]< step_out)
        subptx = pts[idx]
        idx = np.where(subptx[:,2]>=step_in)
        test = subptx[idx]
        x = test[:,0].mean()*3.0
        y = test[:,1].mean()*3.0
        z = (step_in + step_out)/2.0
        path.append([x,y,z])
    return path

def animation_callback(vis):
    
    global pcd_list
    if len(pcd_list) >0 :
        

        #print("updated - Queue count = ", len(pcd_list))

        next_pcd = pcd_list.pop(0)

        if SHOW_PCD:
            global pcd
            pcd.colors = next_pcd.colors
            pcd.points = next_pcd.points 
            vis.update_geometry(pcd)
        #axis_set.colors=o3d.utility.Vector3dVector(np.float64(axis_color))

        if SHOW_FOV:
            next_fov = fov_set_list.pop(0)
            global fov_set
            fov_set.points = next_fov.points
            fov_set.lines = next_fov.lines
            vis.update_geometry(fov_set)

        if len(axis_set_list)>0:
            next_axis = axis_set_list.pop(0)
            global axis_set
            axis_set.points = next_axis.points
            axis_set.lines = next_axis.lines
            axis = np.asarray(next_axis.points)

            if SHOW_NAVI_TRIANGLE:            
                width = 0.15
                depth = 0.02
                height = 0.02

                pos_x = 0
                pos_y = 0

                triangle_vertices=[]
                triangle_triangles = [] 
                scale = 3.0

                global triangle_mesh_set            
                triangle_mesh_set.paint_uniform_color([1,0.706,0])
                triangle_mesh_set.vertices = o3d.utility.Vector3dVector(triangle_vertices)
                triangle_mesh_set.triangles = o3d.utility.Vector3iVector(triangle_triangles)
                vis.update_geometry(triangle_mesh_set)

        if SHOW_AXIS:            
            global point_sphere
            global prev_axis

            axis_vertices = []
            value = -axis[len(axis)-1, 2]
            if(len(prev_axis)>0):
                for i in range(len(axis)):
                    axis[i,0] = (axis[i,0] + prev_axis[i,0])/2.0
                    axis[i,1] = (axis[i,1] + prev_axis[i,1])/2.0
                    axis[i,2] = 0
            axis_set.points = o3d.utility.Vector3dVector(axis)


            point_sphere.translate( [axis[len(axis)-1, 0],axis[len(axis)-1, 1],0 ], relative = False)
            axis_color = [[i/len(axis), i/len(axis), i/len(axis)] for i in range(len(axis))]

            vis.update_geometry(point_sphere)
            vis.update_geometry(axis_set)

            global line_mesh
            line_mesh.remove_line(vis)
            line_mesh.update_line_mesh(axis,np.asarray(axis_set.lines),np.asarray(axis_set.colors), radius=0.005)
            line_mesh.add_line(vis)                
            prev_axis = axis

        if SHOW_TRAJECTORY:
            next_trajectory = trajectory_list.pop(0)
            global trajectory_set
            trajectory_set.points = next_trajectory.points 
            trajectory_set.lines = next_trajectory.lines 
            vis.update_geometry(trajectory_set)

        if SHOW_VIEW_PCD:
            next_view_pcd = view_pcd_list.pop(0)
            global view_pcd_set
            view_pcd_set.colors = next_view_pcd.colors
            view_pcd_set.points = next_view_pcd.points
            vis.update_geometry(view_pcd_set)

        render = vis.get_render_option()
        render.point_size = 4.0
        render.line_width = 0.01       
        vis.update_renderer()
        vis.poll_events()

    else:
        time.sleep(0.03)

def update_path():
    global i
    dirname = "./media/image_"
    filenames_image = os.listdir(dirname)
    filenames_image.sort()

    #dirname_depth = "/home/sj/src/open3d/"+ d_path+"_result"
    dirname_depth = "./media/gt_"
    filenames_depth = os.listdir(dirname_depth)
    filenames_depth.sort()  

    full_image_filename = os.path.join(dirname, filenames_image[i])
    full_depth_filename = os.path.join(dirname_depth, filenames_depth[i])
    i=i+1
    print(i, full_image_filename)
    return full_image_filename, full_depth_filename

def load_image_my(image_path, depth_path):
    color_raw = o3d.io.read_image(image_path)
    pp =  np.concatenate((np.array(color_raw)[:,:,0].reshape(256,256,1), np.array(color_raw)[:,:,1].reshape(256,256,1), np.array(color_raw)[:,:,2].reshape(256,256,1)),axis=2)
    color_raw = o3d.geometry.Image(pp)
    depth_raw = o3d.io.read_image(depth_path)
    return color_raw, depth_raw

def set_fov_line():
    fov_center = [0,0,0]
    near = 0.12
    far = 0.5
    width = 0.25
    height = 0.25
    ratio = near/far

    fov_near_lt = [-width*ratio, height*ratio, near]
    fov_near_lb = [-width*ratio, -height*ratio, near]
    fov_near_rt = [width*ratio, height*ratio, near]
    fov_near_rb = [width*ratio, -height*ratio, near]

    fov_far_lt = [-width, height, far]
    fov_far_lb = [-width, -height, far]
    fov_far_rt = [width, height, far]
    fov_far_rb = [width, -height, far]

    fov = [ fov_near_lt, fov_near_lb,fov_near_rb , fov_near_rt, fov_far_lt,fov_far_lb,fov_far_rb,fov_far_rt, fov_center]
    fov_lines = [[0,1], [1,2],[2,3], [0,3], [4,5],[5,6],[6,7],[4,7],[4,8],[5,8],[6,8],[7,8]] 
    fov_color = [[0,0,1], [0,0,1], [0,0,1], [0,0,1], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0], [0,1,0]]
    return fov, fov_lines, fov_color


def run_vis():
    global timer2
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()


    if timer2 %100 ==0:
        source_color,source_depth = update_path()
    timer2+=1
    print(timer2)

    source_color_raw, source_depth_raw = load_image_my(source_color,source_depth )
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color_raw, source_depth_raw,convert_rgb_to_intensity=False)

    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    #pinhole_camera_intrinsic.set_intrinsics(256,256,200.48,250.4,128,128)
    pinhole_camera_intrinsic.set_intrinsics(256,256,15.78,15.72,128,128)



    global pcd
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    pts = np.asarray(pcd.points)

    if SHOW_PCD:
        vis.add_geometry(pcd)

    #path 
    axis = path_loader_inv(pts)
    lines = [[i, i+1] for i in range(len(axis)-1)]
    axis_color = [[i/len(lines), 0, 1-i/len(lines)] for i in range(len(lines))]
    print(axis, lines)
    global axis_set
    axis_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axis),
        lines=o3d.utility.Vector2iVector(lines)
    )
    
    if SHOW_AXIS:            
        global point_sphere
        point_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01, resolution=20)
        point_sphere.paint_uniform_color([1,0.0,0])
        vis.add_geometry(point_sphere)
        axis_set.colors=o3d.utility.Vector3dVector(np.float64(axis_color))
        vis.add_geometry(axis_set)

        global line_mesh
        axis = np.array(axis)
        line_mesh = LineMesh(axis,lines,axis_color, radius=0.005)
        line_mesh.add_line(vis)



    #trajectory 
    if SHOW_TRAJECTORY:
        trajectory = [[0, 0, 0], [0,0,-0.1]]
        trajectory_lines = [[i, i+1] for i in range(len(trajectory)-1)]    
        global trajectory_set
        trajectory_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(trajectory),
            lines=o3d.utility.Vector2iVector(trajectory_lines)
        )        
        vis.add_geometry(trajectory_set)

    if SHOW_FOV:
        #FOV area
        fov, fov_lines, fov_color = set_fov_line()
        global fov_set
        fov_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(fov),
            lines=o3d.utility.Vector2iVector(fov_lines)
        )    
        fov_set.colors=o3d.utility.Vector3dVector(np.float64(fov_color))
        vis.add_geometry(fov_set)

    if SHOW_VIEW_PCD:
        global view_pcd_set
        view_pcd_set = o3d.geometry.PointCloud.create_from_rgbd_image(
                source_rgbd_image, pinhole_camera_intrinsic)
        view_pcd_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        vis.add_geometry(view_pcd_set)

    # base = [ [0.005,0.005,0], [ -0.005,-0.005,0], [-0.005,0.005,0.0],[ -0.005,-0.005,-0.005] ]
    # base_lines = [[1,3], [0,2],[1,2]] 
    # base_color = [[0,1,1],[1,0,0],[0,1,0]]
    base = [ [0.75,0.75,0], [ -0.75,-0.75,0], [-0.75,0.75,0.0],[ -0.75,-0.75,-0.75] ]
    base_lines = [[1,3], [0,2],[1,2]] 
    base_color = [[0,1,1],[1,0,0],[0,1,0]]
    #base 
    # base = [ [0,0,0], [ 0,1.0,0], [1.0,0,0], [0,0,-1] ]
    # base_lines = [[0,1], [0,2],[0,3]] 
    # base_color = [[0,1,1],[1,0,0],[0,1,0]]
    
    base_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(base),
        lines=o3d.utility.Vector2iVector(base_lines)
    )    
    base_set.colors=o3d.utility.Vector3dVector(np.float64(base_color))

    vis.add_geometry(base_set)
    #mesh_box = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.2, cone_radius=0.2, cylinder_height=0.002, cone_height=0.2, resolution=20, cylinder_split=4, cone_split=1)
    
    if SHOW_NAVI_TRIANGLE:
        triangle_vertices = [[0,0,0], [0,0,0]]
        triangle_lines = [[0,1]]     
        triangle_color = [[0,1,0] ]
        global triangle_mesh_set
        triangle_mesh_set = o3d.geometry.LineSet(vertices=o3d.utility.Vector3dVector(triangle_vertices), triangles=o3d.utility.Vector2iVector(triangle_triangles))
        triangle_mesh_set.colors = o3d.utility.Vector3dVector(np.float64(triangle_color))

        # triangle_vertices = [ [-0.1,0,0], [ 0.1,0,0], [0,0.2,-0.1],   [-0.1+0.4,0,0], [ 0.1+0.4,0,0], [0+0.4,0.2,-0.1] , [-0.1+0.7,0,-0.2], [ 0.1+0.7,0,-0.2], [0+0.7,0.2,-0.3]]
        # triangle_triangles = [[0,1,2], [3,4,5], [6,7,8]]              
        # triangle_mesh.vertices = o3d.utility.Vector3dVector(triangle_vertices)
        # triangle_mesh.triangles = o3d.utility.Vector3iVector(triangle_triangles)

        vis.add_geometry(triangle_mesh_set)    

    #base_color = [[0,1,1],[1,0,0],[0,1,0]]
    #mesh_box = o3d.geometry.TriangleMesh.create_tetrahedron(radius=0.1)
    
    view_ctl = vis.get_view_control()
    #view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
    view_ctl.set_up((0, 1, 0))  # set the negative direction of the y-axis as the up direction
    view_ctl.set_front((0, 0, +0.1))  # set the positive direction of the x-axis toward you
    view_ctl.set_lookat((0, 0, -0.1))  # set the original point as the center point of the window

    render = vis.get_render_option()
    render.point_size = 4.0
    render.line_width = 2.0
    #render.show_coordinate_frame = True
    render.point_show_normal = True
    
    vis.register_animation_callback(animation_callback)
    vis.run()

def update_pose(PrevT, T):
    cam_to_world = np.dot( PrevT , T)
    xyzs = cam_to_world[:3, 3]
    return cam_to_world, xyzs

@torch.no_grad()
def infer_and_vis(input_file, prev_path, output_file, model_wrapper, image_shape, half, save , T, trajectory_lines_list,trajectory_points_list, imagemode):
    
    mode =1 # 0 = gt, 1 = prediction
    mode_colored = 0 # 0 = color, 1 = depth color

    scale = 1
    upscale = nn.UpsamplingBilinear2d(scale_factor=scale)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    if imagemode ==0:
        image = load_image(input_file).convert('RGB')
        prev_image = load_image(prev_path).convert('RGB')
    elif imagemode ==1:
        input_file = cv2.cvtColor(input_file, cv2.COLOR_BGR2RGB)
        image = pil.fromarray(input_file)

        prev_path = cv2.cvtColor(prev_path, cv2.COLOR_BGR2RGB)
        prev_image = pil.fromarray(prev_path)


    w, h = image.size
    width_crop = 210
    height_crop = 40
    image = image.crop((width_crop,height_crop, w-width_crop, h-height_crop))
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Load image
    
    # Resize and to tensor
    prev_image = resize_image(prev_image, image_shape)
    prev_image = to_tensor(prev_image).unsqueeze(0)


    if mode ==0:
        depth_path = input_file.replace("testimage", "dataAD50")
        depth_path = depth_path.replace("FrameBuffer", "scaled")
        gt_depth = load_image(depth_path)
        gt_depth = resize_image(gt_depth, image_shape)
        gt_depth = to_tensor(gt_depth)
        gt_median = gt_depth.median()


    # gt_depth_np = gt_depth.view(256,256,1).numpy()*255

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)
        prev_image = prev_image.to('cuda:{}'.format(rank()), dtype=dtype)
        if mode ==0:
            gt_depth = gt_depth.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)[0]
    _, depth = disp_to_depth(pred_inv_depth, min_depth=0.1, max_depth=255.0)
    depth = depth *0.3
    depth_median = depth.median()

    if mode ==0:
        depth = depth * (gt_median/depth_median)
    # depth_np = depth.permute(1,2,0).detach().cpu().numpy()
    # colored = cv2.applyColorMap(depth_np.astype(np.uint8), cv2.COLORMAP_JET)
    #     rgb = image[0].permute(1, 2, 0).detach().cpu().numpy()*255

    pose = model_wrapper.pose(image,prev_image)
    transPose = [Pose.from_vec(pose[:, i], 'euler') for i in range(pose.shape[1])]



    #camera intrinsic matrix
    K = np.array([[0.646, 0, 0.5, 0],
                [0, 0.6543, 0.5, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)

    #camera intrinsic matrix scaling.float()
    K[0, :] *= 256//2**0
    K[1, :] *= 256//2**0

    K = K[0:3,0:3]
    K = to_tensor(K).to('cuda')

    tcw = Pose.identity(len(K)).to('cuda')
    cam = Camera(K=K, Tcw = tcw)


    #upsample depth

    #################3
    if mode==0:
        world_points, view_world_points = cam.reconstruct2(gt_depth, frame='w')
    else:
        world_points, view_world_points = cam.reconstruct2(depth, frame='w')

    #ref_cam = Camera(K=K, Tcw = transPose[0])
    #transform_points = ref_cam.transform_pose(world_points)

    #world_points = world_points.view(1,1,256,256)
    world_points = upscale(world_points)
    world_points = world_points.view(1,3,256*scale,256*scale)    
    cam_points = world_points.view(3,-1).cpu().numpy()

    view_world_points = upscale(view_world_points)
    view_world_points = view_world_points.view(1,3,256*scale,256*scale)    
    view_world_points[:,2,:,:] = 0.5
    view_pints = view_world_points.view(3,-1).cpu().numpy()


    
    #z= z.view(1,1, 256, 256)
    # sampling_depth = funct.grid_sample(z, xy,  mode='bilinear',padding_mode='zeros', align_corners=True)
    # #_, sampling_depth= disp_to_depth(sampling_depth, min_depth=0.1, max_depth=100.0)
    # sampling_depth= sampling_depth.view(1,256,256)
    # sampling_depth_np = sampling_depth.permute(1,2,0).detach().cpu().numpy()*255
    #sampling_depth_np.save("test.png")
    #imwrite("sampled_depth.png",sampling_depth_np )
    #imwrite("depth.png",depth_np )
    # z to 256,256 

    # grid sample z  using xy 
    pts = np.transpose(cam_points)            
    pts = np.float64(pts)

    #openc3d geometry format
    next_pcd = o3d.geometry.PointCloud()
    next_pcd.points = o3d.utility.Vector3dVector(pts)
    

    pts_view = np.transpose(view_pints)            
    pts_view = np.float64(pts_view)

    #openc3d geometry format
    next_view_pcd = o3d.geometry.PointCloud()
    next_view_pcd.points = o3d.utility.Vector3dVector(pts_view)

    image = upscale(image)
    size = 256*256*scale*scale
    image = image.view(-1,size).permute(1,0).cpu().numpy()
    image = np.float64(image)            


    if mode==0:
        depth = upscale(gt_depth.unsqueeze(0))[0]
    else:
        depth = upscale(depth.unsqueeze(0))[0]
    depth_np = depth.permute(1,2,0).detach().cpu().numpy()*255
    colored = cv2.applyColorMap(depth_np.astype(np.uint8), cv2.COLORMAP_JET)

    #coloerd = upscale(coloerd)Z
    colored = colored.reshape(colored.shape[0]*colored.shape[1],3)
    colored = np.float64(colored)
    colored = colored[...,::-1]/255.0

    if mode_colored==0:
        color = o3d.utility.Vector3dVector(image)
    else:
        color = o3d.utility.Vector3dVector(colored)

    next_pcd.colors = color
    next_view_pcd.colors = color

    tt = transPose[0].to('cpu').item().numpy()[0]
    
    tt[0,3] = (tt[0,3])
    tt[1,3] = (tt[1,3])
    tt[2,3] = (tt[2,3])

    T, xyzs = update_pose(T, tt)
    center = xyzs[0:3]
    x = xyzs[0]
    y = xyzs[1]
    z = xyzs[2]

    if USE_POSE_PREDICT:
        next_pcd.transform(T)        
        next_view_pcd.transform(T)        
    else:
        next_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        next_view_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        

    #path 
    #if SHOW_AXIS:
    pts = np.asarray(next_pcd.points)
    axis = path_loader_inv(pts)
    lines = [[i, i+1] for i in range(len(axis)-1)]
    axis_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(axis),
        lines=o3d.utility.Vector2iVector(lines),
    )
    axis_set_list.append(axis_set)
    
    #trajectory
    if SHOW_TRAJECTORY:
        trajectory_points_list.append([x,y,z])
        trajectory_lines_list.append([len(trajectory_points_list)-2,len(trajectory_points_list)-1])
        trajectory = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(trajectory_points_list),
            lines=o3d.utility.Vector2iVector(trajectory_lines_list),
        )
        #global trajectory_list
        trajectory_list.append(trajectory)


    #fovq
    if SHOW_FOV:
        fov_points, fov_lines, _ = set_fov_line()
        fov = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(fov_points),
            lines=o3d.utility.Vector2iVector(fov_lines),
        )
        if USE_POSE_PREDICT:
            fov.transform(T)
        else:
            fov.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        fov_set_list.append(fov)    


    if SHOW_VIEW_PCD:
        view_pcd_list.append(next_view_pcd)

    #add global list (Queue) (pcd, path, trajectory)
    #global pcd_list
    pcd_list.append(next_pcd)        
    return T
    
@torch.no_grad()
def infer_and_save_depth(input_file, output_file, model_wrapper, image_shape, half, save):

    if not is_image(output_file):
        # If not an image, assume it's a folder and append the input name
        os.makedirs(output_file, exist_ok=True)
        output_file = os.path.join(output_file, os.path.basename(input_file))

    # change to half precision for evaluation if requested
    dtype = torch.float16 if half else None

    # Load image
    image = load_image(input_file).convert('RGB')
    # Resize and to tensor
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)

    # Send image to GPU if available
    if torch.cuda.is_available():
        image = image.to('cuda:{}'.format(rank()), dtype=dtype)

    # Depth inference (returns predicted inverse depth)
    pred_inv_depth = model_wrapper.depth(image)[0]
    #_, depth = disp_to_depth(pred_inv_depth, min_depth=0.1, max_depth=100.0)
    depth = 1/pred_inv_depth
    #d = inv2depth(pred_inv_depth)
    if save == 'npz' or save == 'png':
        # Get depth from predicted depth map and save to different formats
        filename = '{}.{}'.format(os.path.splitext(output_file)[0], save)
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(filename, 'magenta', attrs=['bold'])))
        write_depth(filename, depth=inv2depth(pred_inv_depth))
    else:
        # Prepare RGB image
        depth_np = depth.permute(1,2,0).detach().cpu().numpy()*32
        #depth_np
        rgb = image[0].permute(1, 2, 0).detach().cpu().numpy() * 255
        # Prepare inverse depth
        viz_pred_inv_depth = viz_inv_depth(pred_inv_depth[0]) * 255
        # Concatenate both vertically
        #image = np.concatenate([rgb, viz_pred_inv_depth], 0)
        # Save visualization
        print('Saving {} to {}'.format(
            pcolor(input_file, 'cyan', attrs=['bold']),
            pcolor(output_file, 'magenta', attrs=['bold'])))
        #imwrite(output_file, depth[:, :, ::-1])
        imwrite(output_file, depth_np)
    return pred_inv_depth

def main(args):

    # Initialize horovod
    hvd_init()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape

    # Set debug if requested
    set_debug(config.debug)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # change to half precision for evaluation if requested
    dtype = torch.float16 if args.half else None

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda:{}'.format(rank()), dtype=dtype)

    # Set to eval mode
    model_wrapper.eval()  
    
    t = threading.Thread(target=run_vis)
    t.start()

   # while True:
    global pcd_list, axis_set_list, trajectory_list, fov_set_list, view_pcd_list
    pcd_list = []
    view_pcd_list = []
    axis_set_list = []
    trajectory_list = []
    fov_set_list =[]
    trajectory_lines_list = [[0,1]]
    trajectory_points_list = [[0,0,0],[0,0,0]]
    init_pose = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    T = init_pose

    
    if args.input is not None:
        if os.path.isdir(args.input):
            # If input file is a folder, search for image files
            files = []
            for ext in ['png', 'jpg']:
                files.extend(glob((os.path.join(args.input, '*.{}'.format(ext)))))
            files.sort()
            print0('Found {} files'.format(len(files)))

            for idx, image_path in enumerate(files):
                if idx ==0:
                    continue
                prev_path = files[idx-1]
                T = infer_and_vis(image_path,prev_path, args.output, model_wrapper, image_shape, args.half, args.save, T,trajectory_lines_list,trajectory_points_list, 0 )

        elif os.path.isfile(args.input):
            cap=cv2.VideoCapture(args.input)
            prev_image = 0
            while(cap.isOpened()):
                ret, input_image = cap.read()
                if ret:
                    if prev_image is not 0:
                        T = infer_and_vis(input_image,prev_image, args.output, model_wrapper, image_shape, args.half, args.save, T,trajectory_lines_list,trajectory_points_list, 1)
                    prev_image = input_image
        elif args.input=='cam':
            cap=cv2.VideoCapture(0)
            prev_image = 0
            while(cap.isOpened()):
                ret, input_image = cap.read()
                if ret:
                    if prev_image is not 0:
                        T = infer_and_vis(input_image,prev_image, args.output, model_wrapper, image_shape, args.half, args.save, T,trajectory_lines_list,trajectory_points_list, 1)
                    prev_image = input_image

        else:
            # Otherwise, use it as is
            files = [args.input]
    

if __name__ == '__main__':
    args = parse_args()
    main(args)
