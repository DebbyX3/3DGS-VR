# REMEMBER:
# conda activate 3DGS-VR

import open3d as o3d
import csv
import numpy as np
from numpy.linalg import inv

# ***************** READ IMAGES.TXT COLMAP FILE 

# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)

'''
The reconstructed pose of an image is specified as the projection from world to 
the camera coordinate system of an image using a quaternion (QW, QX, QY, QZ) and 
a translation vector (TX, TY, TZ). The quaternion is defined using the Hamilton 
convention, which is, for example, also used by the Eigen library. 

The coordinates of the projection/camera center are given by -R^t * T, where 
R^t is the inverse/transpose of the 3x3 rotation matrix composed from the 
quaternion and T is the translation vector. 

The local camera coordinate system 
of an image is defined in a way that the X axis points to the right, the Y axis 
to the bottom, and the Z axis to the front as seen from the image.
'''

count = 0
images_info = []
cameras_coords = []

with open('../colmap_reconstructions/colmap_output_simple_radial/sparse/images.txt', 'r') as f:
    for line in f:    
        # Ignore comments
        if not line.startswith("#"):
            count+=1

            if count % 2 != 0: # Read every other line (skip the second line for every image)
                single_img_info = line.split() # split every field in line
                images_info.append(single_img_info) # and store them as separate fields as list in a list ( [ [] ] )

                # Images info contains:
                # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                # 0         1   2   3   4   5   6   7   8          9

                # The coordinates of the projection/camera center are given by -R^t * T, where 
                # R^t is the inverse/transpose of the 3x3 rotation matrix composed from the 
                # quaternion and T is the translation vector.

                # CREATE ROTATION MATRIX 'R' FROM QUATERNIONS
                quaternions = np.array([single_img_info[1], single_img_info[2], single_img_info[3], single_img_info[4]]) # numpy array
                print("Quaternions coeff:", quaternions)
                print("Quaternions shape:", quaternions.shape)

                rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternions)
                print("Rotation matrix from quaternions:\n", rotation_matrix)

                # TRANSPOSE R (R^t)
                rotation_transpose = rotation_matrix.transpose()
                print("Rotation matrix transposed:\n", rotation_transpose)

                # MULTIPLY R_TRANSPOSED BY * (-1) (-R^t)
                rotation_trans_inv = (-1) * rotation_transpose
                print("Rotation matrix transposed and * (-1):\n", rotation_trans_inv)

                # CREATE TRANSLATION VECTOR T
                translation = np.array([single_img_info[5], single_img_info[6], single_img_info[7]], dtype = float)
                print("Translation vector:\n", translation)

                # DOT PRODUCT (*) BETWEEN INVERTED_R_TRANSPOSED (-R^t) AND TRANSLATION VECTOR (T)
                # TO FIND CAMERA CENTER
                camera_center = np.dot(rotation_trans_inv, translation)
                print("Camera center coords:\n", camera_center)

                cameras_coords.append(camera_center)

                print("\n")

# **************** READ POINTS3D.TXT TO PLOT COLMAP SPARSE POINTS

# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)

points_info = []
points_position = []
points_color = []

with open('../colmap_reconstructions/colmap_output_simple_radial/sparse/points3D.txt', 'r') as f:
    for line in f:
        # Ignore comments
        if not line.startswith("#"):
            single_point_info = line.split() # split every field in line
            points_info.append(single_point_info)# and store them as separate fields as list in a list ( [ [] ] )

            # Points info contains:
            # POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
            # 0           1  2  3  4  5  6  7      8

            points_position.append(np.array([single_point_info[1], single_point_info[2], single_point_info[3]], dtype = float))
            points_color.append(np.array([single_point_info[4], single_point_info[5], single_point_info[6]], dtype = float))


# Create new point cloud, add colmap 3D points and colors
colmap_points3d_point_cloud = o3d.geometry.PointCloud()
colmap_points3d_point_cloud.points = o3d.utility.Vector3dVector(points_position)
colmap_points3d_point_cloud.colors = o3d.utility.Vector3dVector((np.array(points_color)) / 255.0)

# ***************** LOAD POINT CLOUD FILE (.PLY)

# Load colmap point cloud   
pcd = o3d.io.read_point_cloud("../colmap_reconstructions/colmap_output_simple_radial/dense/fused.ply") 

# Create new point cloud, add camera centers
cameras_point_cloud = o3d.geometry.PointCloud()
cameras_point_cloud.points = o3d.utility.Vector3dVector(cameras_coords)

# Paint them red
cameras_point_cloud.paint_uniform_color([1, 0, 0])

# View both camera and ply points
#o3d.visualization.draw_geometries([cameras_point_cloud])

# View both camera and colmap points
o3d.visualization.draw([colmap_points3d_point_cloud, cameras_point_cloud])