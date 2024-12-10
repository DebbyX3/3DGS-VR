import open3d as o3d
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
from collections import defaultdict

'''mi pare che il codice nella mappatura pixel-> texel vada fixato semplicemente togliendo la depth. 
Non devi calcolare "punti" se sulle immagini fai loop sui pixel, ti basta ricavare la direzione del 
pixel ray, ruotarla nel riferimento globale e usare il vettore ricavato per il calcolo delle coordinate 
sulla texture. Però penso sia il caso di implementare la procedura inversa dello pseudocodice per cui 
occorre ricavare la funzione che dà colore e  attributo per lo z buffer dati camera e coordinata del 
texel da riempire. Non dovrebbe essere complicato, si può fare anche in quel caso volendo il loop sui 
file delle immagini e dentro fare il loop sui texel ricavare le coordinate immagine e se sono nel range, 
ricavare il colore e la metrica di scelta
naturalmente si escluderanno poi i punti "vicini"'''

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

'''
cameraTxt_path = '../colmap_reconstructions/cavignal-fountain_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../colmap_reconstructions/cavignal-fountain_pinhole_1camera/sparse/images.txt'
imgs_folder = "../colmap_reconstructions/cavignal-fountain_pinhole_1camera/dense/images"
depth_map_folder = '../colmap_reconstructions/cavignal-fountain_pinhole_1camera/dense/stereo/depth_maps'
'''

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
cameras_coords = []

lineset = o3d.geometry.LineSet()
all_points = []
all_lines = []

with open(imagesTxt_path, 'r') as f:
    for line in f:    
        # Ignore comments
        if not line.startswith("#"):
            count+=1

            if(count > 0):
                if count % 2 != 0: # Read every other line (skip the second line for every image)
                    if count % 1 == 0: # salta tot righe
                        
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

                        # ----------- FIND CAMERA CENTER
                        # TRANSPOSE R (R^t)
                        rotation_transpose = rotation_matrix.transpose()

                        # MULTIPLY R_TRANSPOSED BY * (-1) (-R^t)
                        rotation_trans_inv = (-1) * rotation_transpose

                        # CREATE TRANSLATION VECTOR T
                        translation = np.array([single_camera_info[5], single_camera_info[6], single_camera_info[7]], dtype = float)

                        # DOT PRODUCT (*) BETWEEN INVERTED_R_TRANSPOSED (-R^t) AND TRANSLATION VECTOR (T)
                        # TO FIND CAMERA CENTER
                        camera_center = np.dot(rotation_trans_inv, translation)
                        
                        cameras_coords.append(camera_center)

                        ''' mio test'''
                        
                        # Extract camera direction vector (forward vector)
                        # rotation_matrix = extrinsics_matrix[:3, :3]  # I already have the rot matrix, keep it commented
                        forward_vector = -rotation_matrix[:, 2]
                        print(forward_vector)
                    
                        # cerca il punto finale per fare sta linea
                        # punto di inizio è la camera stessa
                        # punto finale = punto inizio + direzione * lunghezza vettore
                        final_point = camera_center + forward_vector * 1.5

                        # ora traccio linea
                        all_points.append(camera_center)
                        all_points.append(final_point)

                        # Aggiungi la linea tra gli ultimi due punti aggiunti
                        idx = len(all_points)
                        all_lines.append([idx - 2, idx - 1])  # Indici degli ultimi due punti


                        ''' fine mio test'''


                        # Take the image file name
                        img_filename = single_camera_info[9]

                        # Read the image
                        img_path = os.path.join(imgs_folder, img_filename)
                        img = np.asarray(Image.open(img_path))   

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

'''
# View the image
plt.imshow(equirectangular_image)
plt.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()
'''

# Create new point cloud, add camera centers
cameras_point_cloud = o3d.geometry.PointCloud()
cameras_point_cloud.points = o3d.utility.Vector3dVector(cameras_coords)

# Paint them red
cameras_point_cloud.paint_uniform_color([1, 0, 0])

lineset.points = o3d.utility.Vector3dVector(all_points)
lineset.lines = o3d.utility.Vector2iVector(all_lines)

# Apply color to lineset
GREEN = [0.0, 1.0, 0.0]
lines_color = [GREEN] * len(lineset.lines)
lineset.colors = o3d.utility.Vector3dVector(lines_color)

o3d.visualization.draw_geometries([lineset, cameras_point_cloud])