import open3d as o3d
import numpy as np
from PIL import Image
import os

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(
            fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
        )
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


# ************************** EXTRACT INTRINSICS FROM CAMERA.TXT FILE **************************
# Intrinsics matrix:
# [fx, 0, cx]
# [0, fy, cy]
# [0, 0,  1 ]


# ************** READ COLMAP CAMERA.TXT FILE    

# ***  WARNING: THIS SCRIPT ASSUMES THAT ALL CAMERAS HAVE THE SAME INTRINSICS ***
# ***  SO IN THE CAMERA.TXT FILE WE WILL ONLY READ THE FIRST CAMERA INTRINSICS ***
# *** (ALSO BEACUSE THERE IS ONLY ONE CAMERA IN THE CAMERA.TXT FILE IF THEY SHARE THE SAME INTRINSICS) ***

# Camera list with one line of data per camera:
#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
#
# In case of Pinhole camera model (example):
# 1 PINHOLE 3072 2304 2560.56 2560.56 1536 1152
# 
# In case of Simple Pinhole camera model (example):
# 2 SIMPLE_PINHOLE 3072 2304 2559.81 1536 1152
#
# In case of Simple Radial camera model (example):
# 3 SIMPLE_RADIAL 3072 2304 2559.69 1536 1152 -0.0218531

camera_info = []

# Load the camera intrinsics 
with open('../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/cameras.txt', 'r') as f:
    for line in f:    
        # Ignore comments
        if not line.startswith("#"):
            single_camera_info = line.split() # split every field in line
            camera_info.append(single_camera_info) # and store them as separate fields as list in a list ( [ [] ] )

            # Camera info contains:
            # CAMERA_ID  MODEL   WIDTH   HEIGHT  PARAMS[]
            # 0          1       2       3       4   5   6   7   8
            # Where PARAMS[] are:
            # SIMPLE_PINHOLE: fx (fx = fy), cx, cy      1 focal length and principal point
            # PINHOLE: fx, fy, cx, cy                   2 focal lenghts and principal point
            # SIMPLE_RADIAL: fx (fx = fy), cx, cy, k1   1 focal length, principal point and radial distortion
            # RADIAL: fx (fx = fy), cx, cy, k1, k2      1 focal lengths, principal point and 2 radial distortions

            width = int(single_camera_info[2])
            height = int(single_camera_info[3])

            if single_camera_info[1] == "SIMPLE_PINHOLE":
                fx = float(single_camera_info[4])
                fy = float(single_camera_info[4]) #same as fx
                cx = float(single_camera_info[5])
                cy = float(single_camera_info[6])

            if single_camera_info[1] == "PINHOLE":
                fx = float(single_camera_info[4])
                fy = float(single_camera_info[5]) 
                cx = float(single_camera_info[6])
                cy = float(single_camera_info[7])

            if single_camera_info[1] == "SIMPLE_RADIAL":
                fx = float(single_camera_info[4])
                fy = float(single_camera_info[4]) #same as fx
                cx = float(single_camera_info[5])
                cy = float(single_camera_info[6])
                k1 = float(single_camera_info[7])

            if single_camera_info[1] == "RADIAL":
                fx = float(single_camera_info[4])
                fy = float(single_camera_info[4]) #same as fx
                cx = float(single_camera_info[5])
                cy = float(single_camera_info[6])
                k1 = float(single_camera_info[7])
                k2 = float(single_camera_info[8])  

            break    # We only need the first camera intrinsics (assume all cameras have the same intrinsics)  

# Create the camera intrinsic matrix
intrinsic_matrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])

print("--- Camera: ", camera_info[0][1])
print(" Width: ", width)
print(" Height: ", height)
print(" fx: ", fx)
print(" fy: ", fy)
print(" cx: ", cx)
print(" cy: ", cy)  

if 'k1' in locals():
    print(" k1: ", k1)
if 'k2' in locals():
    print(" k2: ", k2)

#intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix)


# ************************** EXTRACT EXTRINSICS **************************

# Extrinsics matrix:
# [R 3x3, T 3x1]
# [0 1x3, 1    ] 4x4
#
# Which means:
# [ R11 R12 R13 T1 ]
# [ R21 R22 R23 T2 ]
# [ R31 R32 R33 T3 ]
# [ 0   0   0   1  ]
#
# Where R is the 3x3 rotation matrix and T is the translation vector
# The matrix denote the coordinate system transformations from 3D world coordinates to 3D camera coordinates

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
cameras_info = []
cameras_extrinsics = []

# Create the point cloud
point_cloud = o3d.geometry.PointCloud()

with open('../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/images.txt', 'r') as f:
    for line in f:    
        # Ignore comments
        if not line.startswith("#"):
            count+=1
            print(count)

            if(count > 0):

                if count % 2 != 0: # Read every other line (skip the second line for every image)
                    single_camera_info = line.split() # split every field in line
                    cameras_info.append(single_camera_info) # and store them as separate fields as list in a list ( [ [] ] )

                    # Images info contains:
                    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                    # 0         1   2   3   4   5   6   7   8          9

                    # CREATE ROTATION MATRIX 'R' FROM QUATERNIONS
                    quaternions = np.array([single_camera_info[1], single_camera_info[2], single_camera_info[3], single_camera_info[4]]) # numpy array
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternions)
                    #print("Rotation matrix from quaternions:\n", rotation_matrix)

                    # CREATE TRANSLATION VECTOR T
                    translation = np.array([single_camera_info[5], single_camera_info[6], single_camera_info[7]], dtype = float)
                    #print("Translation vector:\n", translation)

                    # CREATE EXTRINSICS MATRIX                
                    extrinsics_matrix = np.vstack([np.hstack([rotation_matrix, translation.reshape(3, 1)]), 
                                                    np.array([0, 0, 0, 1])])
                    print("Extrinsics matrix:\n", extrinsics_matrix)

                    cameras_extrinsics.append(extrinsics_matrix)
                    
                    # Take the image file name
                    img_filename = single_camera_info[9]

                    # Search the image file name in the depth map directory
                    depth_map_folder = '../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/stereo/depth_maps'
                    depth_map_filename = img_filename + '.geometric.bin'
                    depth_map_path = os.path.join(depth_map_folder, depth_map_filename)

                    # Load the depth map image
                    #depth_image = Image.open("path_to_depth_image.png")
                    #depth_array = np.array(depth_image)

                    depth_map = read_array(depth_map_path)
                    #depth_map = np.load("../colmap_reconstructions/colmap_output_simple_radial/dense/stereo/depth_maps_npy_colmap/depth_map_000001.png.geometric.bin.npy")

                    depth_map_o3d = o3d.geometry.Image(depth_map) 

                    # ***** COLORS

                    # Load the RGB image
                    rgb_image = o3d.io.read_image("../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/images/" + img_filename)

                    # Extract the RGB values of every pixel
                    rgb_values = np.asarray(rgb_image).reshape(-1, 3)

                    # ***** CREATE POINT CLOUD

                    frog = (np.array(rgb_values) / 255.0)
                    
                    # Create the point cloud from the depth map
                    single_point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_map_o3d, intrinsic, extrinsic = extrinsics_matrix)
                    
                    # Estrai i colori dall'immagine RGB
                    rgb_values = np.asarray(rgb_image).reshape(-1, 3) / 255.0

                    # Associa i colori alla point cloud
                    single_point_cloud.colors = o3d.utility.Vector3dVector(rgb_values)

                    # Aggiungi alla point cloud totale
                    point_cloud += single_point_cloud

            if (count == 20):
               break

print(len(point_cloud.points))

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])