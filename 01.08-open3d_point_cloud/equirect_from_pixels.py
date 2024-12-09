import open3d as o3d
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
from typing import Union
import time
import random
from collections import defaultdict

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

# Project pixel in 3d using depth
def project_pixels_with_depth(image, depth, K, extrinsics, width, height):
    # img size
    h_img, w_img = image.shape[:2]
    
    # grid with pixel coords
    u, v = np.meshgrid(np.arange(w_img), np.arange(h_img))
    
    # Stack with pixel coords
    pixel_coords = np.stack([u.ravel(), v.ravel(), np.ones_like(u.ravel())], axis=1).T  # (3, N)
    
    # Depth corrispondente per ogni pixel
    # Check for 0 values
    depth_values = depth.ravel()
    valid_mask = depth_values > 0
    pixel_coords = pixel_coords[:, valid_mask]
    depth_values = depth_values[valid_mask]

    # no depth check
    #depth_values = depth.ravel()
    
    # Converti le coordinate dei pixel in 3D nel sistema locale della camera
    d_local = np.linalg.inv(K) @ (pixel_coords * depth_values)  # Moltiplica con depth

    # Transform to global coords using extrinsics
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    p_global = (R @ d_local) + t[:, None]  # (3, N)
    # p_global[0] -> X
    # p_global[1] -> Y
    # p_global[2] -> Z

    # compute spherical coords
    r = np.linalg.norm(p_global, axis=0)
    theta = np.arctan2(p_global[2], p_global[0])
    phi = np.arcsin(p_global[1] / r)

    # map on equirectangular texture coords
    u_texel = ((theta / (2 * np.pi)) * width).astype(np.int32) % width
    v_texel = ((1 - (phi + np.pi / 2) / np.pi) * height).astype(np.int32)

    texel_coords = np.stack([u_texel, v_texel], axis=1)  # (N, 2)
    colors = image[v.ravel(), u.ravel()]

    return texel_coords, colors, depth_values

# ************************** PATHS **************************
cameraTxt_path = '../colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/images.txt'
imgs_folder = "../colmap_reconstructions/water_bottle_gui_pinhole_1camera/images"
depth_map_folder = '../colmap_reconstructions/water_bottle_gui_pinhole_1camera/stereo/depth_maps'

cameraTxt_path = '../colmap_reconstructions/colmap_output_simple_radial/sparse/cameras.txt'
imagesTxt_path = '../colmap_reconstructions/colmap_output_simple_radial/sparse/images.txt'
imgs_folder = "../colmap_reconstructions/colmap_output_simple_radial/dense/images"
depth_map_folder = '../colmap_reconstructions/colmap_output_simple_radial/dense/stereo/depth_maps'

cameraTxt_path = '../colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/images.txt'
imgs_folder = "../colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/images"
depth_map_folder = '../colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/stereo/depth_maps'

cameraTxt_path = '../colmap_reconstructions/cavignal-fountain_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../colmap_reconstructions/cavignal-fountain_pinhole_1camera/sparse/images.txt'
imgs_folder = "../colmap_reconstructions/cavignal-fountain_pinhole_1camera/dense/images"
depth_map_folder = '../colmap_reconstructions/cavignal-fountain_pinhole_1camera/dense/stereo/depth_maps'

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
with open(cameraTxt_path, 'r') as f:
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

#intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy) #alternatively
intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic_matrix)

count = 0
cameras_info = []
cameras_extrinsics = []

point_cloud = o3d.geometry.PointCloud()

# ************************** EXTRACT EXTRINSICS FROM IMAGES.TXT FILE **************************
# Extrinsic matrix:
# [r1.1, r1.2, r1.3, tx]
# [r2.1, r2.2, r2.3, ty]
# [r3.1, r3.2, r3.3, tz]
# [0,    0,    0,    1 ]

width, height = 2048, 1024
texture = defaultdict(list)

with open(imagesTxt_path, 'r') as f:
    for line in f:    
        # Ignore comments
        if not line.startswith("#"):
            count+=1

            if(count > 0):
                if count % 2 != 0: # Read every other line (skip the second line for every image)
                    if count % 191 == 0: # salta tot righe
                        
                        print(count)

                        single_camera_info = line.split() # split every field in line
                        cameras_info.append(single_camera_info) # and store them as separate fields as list in a list ( [ [] ] )

                        # Images info contains:
                        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                        # 0         1   2   3   4   5   6   7   8          9

                        # CREATE ROTATION MATRIX 'R' FROM QUATERNIONS
                        quaternions = np.array([single_camera_info[1], single_camera_info[2], single_camera_info[3], single_camera_info[4]]) # numpy array
                        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternions)

                        # CREATE TRANSLATION VECTOR T
                        translation = np.array([single_camera_info[5], single_camera_info[6], single_camera_info[7]], dtype = float)

                        # CREATE EXTRINSICS MATRIX                
                        extrinsics_matrix = np.vstack([np.hstack([rotation_matrix, translation.reshape(3, 1)]), 
                                                        np.array([0, 0, 0, 1])])

                        cameras_extrinsics.append(extrinsics_matrix)
                                                
                        # Take the image file name
                        img_filename = single_camera_info[9]

                        # Read the image
                        img_path = os.path.join(imgs_folder, img_filename)
                        img = np.asarray(Image.open(img_path))      

                        # Read the depth map
                        depth_map_filename = img_filename + '.geometric.bin' # get the filename of the depth map
                        depth_map_path = os.path.join(depth_map_folder, depth_map_filename)
                        
                        depth = read_array(depth_map_path)

                        # ----- IMAGE TEXTURE
                        #(image, intrinsics, extrinsics)
                        texel_coords, colors, depth_values = project_pixels_with_depth(img, depth, intrinsic_matrix, extrinsics_matrix, width, height)

                        # Add colors and metadata to texels
                        '''
                        # do not include depth
                        for (u_texel, v_texel), color in zip(texel_coords, colors):
                            texture[(u_texel, v_texel)].append({
                                "color": color,
                                "img_id": single_camera_info[0]
                            })
                        '''

                        # include depth
                        for (u_texel, v_texel), color, depth in zip(texel_coords, colors, depth_values):
                            texture[(u_texel, v_texel)].append({
                                "color": color,
                                "depth": depth,
                                "img_id": single_camera_info[0]
                            })

# Post-processing: create texture
equirectangular_image = np.zeros((height, width, 3), dtype=np.uint8)

# BLENDING DI OGNI TEXEL - risultato un po' schifo
'''
for (u_texel, v_texel), pixel_stack in texture.items():
    # Combina i colori (tipo media)
    colors = np.array([p["color"] for p in pixel_stack])
    equirectangular_image[v_texel, u_texel] = np.mean(colors, axis=0).astype(np.uint8)
'''

# PRENDI SOLO IL PIù VICINO IN BASE ALLA DEPTH - depth min in base al texel
for (u_texel, v_texel), pixel_stack in texture.items():
    # in numpy array
    stack_array = np.array([(p["color"], p["depth"]) for p in pixel_stack], dtype=object)
    
    # Extract colors and depth
    colors = np.stack(stack_array[:, 0])  # (N, 3)       
    depths = np.array(stack_array[:, 1], dtype=np.float32)  # (N, 1)
    
    # Find index of pixel with min depth
    min_index = np.argmin(depths)

    # Set texel color
    equirectangular_image[v_texel, u_texel] = colors[min_index]

# View the image
plt.imshow(equirectangular_image)
plt.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()