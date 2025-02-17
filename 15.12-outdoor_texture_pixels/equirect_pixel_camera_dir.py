import open3d as o3d
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
import math
from collections import defaultdict

# FORWARD MAPPING - (project from image to texture) Project pixel in 3d using depth
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

# FORWARD MAPPING - (project from image to texture)
def forward_mapping_project_pixels(image, intrinsics, extrinsics, texture_width, texture_height):
    # img size
    h_img, w_img = image.shape[:2]
    
    # grid with pixel coords
    u, v = np.meshgrid(np.arange(w_img), np.arange(h_img))
    
    # Stack with pixel coords
    pixel_coords = np.stack([u.ravel(), v.ravel(), np.ones_like(u.ravel())], axis=1).T  # (3, N)
    
    # Converti le coordinate dei pixel in 3D nel sistema locale della camera
    '''
    Direction Vector Calculation: The resulting d_local vector represents 
    the direction from the camera center to the pixel in the image. 
    This is useful for tasks such as ray tracing, where we need to know 
    the direction of rays passing through each pixel, or for reconstructing 
    3D scenes from multiple images.'''
    d_local = np.linalg.inv(intrinsics) @ (pixel_coords)

    # Transform to global coords using extrinsics
    R = extrinsics[:3, :3]

    # prova a non sommare la trasl per trovare il forward vector - forward_vector = -rotation_matrix[:, 2]
    # t = extrinsics[:3, 3]
    # p_global = (R @ d_local) + t[:, None]  # (3, N) 
    
    d_global = R @ d_local
    d_global = d_global / np.linalg.norm(d_global, axis=0) # Normalizza il vettore della direzione, poi usa d_global per calcolare theta e phi

    # d_global[0] -> X
    # d_global[1] -> Y
    # d_global[2] -> Z

    # compute spherical coords

    # r diventa la normale di d_global, usata sopra
    # d global è già stato normalizzato usando r, quindi non mi serve per calcolre phi
    # r = np.linalg.norm(d_global, axis=0)      
    theta = np.arctan2(d_global[2], d_global[0])
    phi = np.arcsin(d_global[1]) # ho tolto la div per r perchè ho già normalizzato

    # map on equirectangular texture coords
    # ---- TEST DI NORMALIZZAZIONE
    # IL COD COMMENTATO è QUELLO ORIGINALE - non normalizzato
    '''
    u_texel = ((theta / (2 * np.pi)) * texture_width).astype(np.int32) % texture_width
    v_texel = ((1 - (phi + np.pi / 2) / np.pi) * texture_height).astype(np.int32)
    '''
    # COD NUOVO - normalizzato
    u_texel = ((theta + np.pi) / (2 * np.pi) * texture_width).astype(np.int32) % texture_width
    v_texel = ((phi + np.pi / 2) / np.pi * texture_height).astype(np.int32)

    texel_coords = np.stack([u_texel, v_texel], axis=1)  # (N, 2)
    colors = image[v.ravel(), u.ravel()]

    return texel_coords, colors

# INVERSE MAPPING (from texture/texel to image)
'''
1	For each texel (u, v) in the texture:
    - Compute the direction (θ,ϕ) in the global reference system
    - Transform the direction in the camera reference system to determine the intersection point with the image plane (using extrinsics)
2	Project the point on the image plane by using the intrinsics and check if it falls INSIDE the image frame/limit
3	If true, compute the color and update the texel

For better understanding:
proiezione inversa (wglobal→wlocal→(u,v)):
- trasformo da w_global a w_local
- coordinate immagine (u,v)
- mapping da texel a direzione globale (θ,ϕ)

1- from texel coords (utexel, vtexel) -> to global dir
    Each texel of the equirect texture represents a direction (theta, phi)
    theta = (u / width) * 2 * pi - pi
    phi = (v / height) * pi - pi / 2

2 - from global dir (theta, phi) -> to image coords (u_image, v_image)
    global direction w_global is computed from (theta, phi):
    w_global = [cos(phi) * cos (theta)
                sin(phi)
                cos(phi) * sin(theta)]
    Then, w_global is transfomed in local direction w_local with:
    w_local = R^T * (w_global)     

3 - project on image plane
    local direction d_local is projected on the image plane using the camera intrinsics:
    d_local = intrinsics * w_local       

4 - Crea una maschera di validità per prendere solo i punti davanti la telecamera
    valid_mask = d_local[..., 2] > 0
    
5 - Normalize d_local to obtain image coords (u_image, v_image)
	dividi per Z per ottenere le coordinate immagine (proiezione prospettica)
    u_image = d_local[0] / d_local[2]
    v_image = d_local[1] / d_local[2]

6 - check if the projected point is inside the image frame
    escludi punti fuori dai bordi - no clipping
	valid_mask &= (u_image >= 0) & (u_image < image_width)
    valid_mask &= (v_image >= 0) & (v_image < image_height)

7 - (optional) Do a blending or a z buffer or similar 
'''
def inverse_mapping(texture_width, texture_height, extrinsics, intrinsics, image, image_width, image_height, pixel_distances):   
    # griglia di texel (u_texel, v_texel)
    u_texel, v_texel = np.meshgrid(np.arange(texture_width), np.arange(texture_height))

    # calcolo di (theta, phi) per ogni texel
    theta = (u_texel / texture_width) * 2 * np.pi - np.pi
    phi = (v_texel / texture_height) * np.pi - (np.pi / 2)

    # direzione globale w_global
    # global directions vector (shape: (height, width, 3))
    # the 3 arrays are combined into a single array where each element is a vector of 3 components
    w_global = np.stack([
        np.cos(phi) * np.cos(theta),
        np.sin(phi),
        np.cos(phi) * np.sin(theta)
    ], axis=-1) # -1: take the last axis

    # trasformazione in direzione locale w_local
    '''
    Qui il calcolo trasforma 
    w_global(direzione globale) nel sistema di coordinate della telecamera (locale), utilizzando:
    La matrice di rotazione 
    - R^T  (trasposta della matrice di rotazione della telecamera).
    w_local = R^T * (w_global)
    La trasf è senza traslazione perchè stiamo considerando punti all'infinito! -> la dir globale w_global è semplicemente ruotata nel sist di rif locale della telecamera

    '''
    R = extrinsics[:3, :3]
    w_local = np.einsum('ij,hwj->hwi', R.T, w_global)  # (height, width, 3)

    # proiezione su piano immagine
    d_local = np.einsum('ij,hwj->hwi', intrinsics, w_local)  # (height, width, 3)

    # maschera di validità
    valid_mask = d_local[..., 2] > 0  # solo punti davanti alla telecamera

    # calcolo di u_image e v_image
    #uguale anche con queste due righe sotto commentate
    #u_image = np.full((texture_height, texture_width), -1, dtype=np.int32)
    #v_image = np.full((texture_height, texture_width), -1, dtype=np.int32)
    u_image = np.full_like(u_texel, -1, dtype=int)
    v_image = np.full_like(v_texel, -1, dtype=int)

    # dividi per Z per ottenere le coordinate immagine (proiezione prospettica)
    u_image[valid_mask] = (d_local[valid_mask, 0] / d_local[valid_mask, 2]).astype(int)
    v_image[valid_mask] = (d_local[valid_mask, 1] / d_local[valid_mask, 2]).astype(int)

    # escludi punti fuori dai bordi - no clipping
    valid_mask &= (u_image >= 0) & (u_image < image_width)
    valid_mask &= (v_image >= 0) & (v_image < image_height)


    ''''''# se attivo zbuffer

    # colori della texture proiettati
    u_image_valid = u_image[valid_mask]
    v_image_valid = v_image[valid_mask]

    # estrai i colori dall'immagine originale per i texel validi
    colors = np.zeros((texture_height, texture_width, 3), dtype=np.uint8)
    colors[valid_mask] = image[v_image_valid, u_image_valid]

    # Mappa le distanze dall'immagine originale ai texel validi
    distances_texture = np.full((texture_height, texture_width), -np.inf)
    distances_texture[valid_mask] = pixel_distances[v_image[valid_mask], u_image[valid_mask]]

    return colors, valid_mask, distances_texture
    

    '''
    ''''''# se attivo blending
    # blending
    colors = np.zeros((height, width, 3), dtype=np.uint8)
    for u, v in zip(u_image[valid_mask], v_image[valid_mask]):
        if colors[v, u].sum() == 0:  # Non c'è ancora alcun contributo in questo texel
            colors[v, u] = image[v, u]
        else:
            colors[v, u] = (colors[v, u] + image[v, u]) / 2
    
    return colors, valid_mask   
    '''

def calculate_pixel_distances_from_camera_center(intrinsics, image_width, image_height):
    # 1. Crea una griglia di coordinate pixel (u, v)
    u, v = np.meshgrid(np.arange(image_width), np.arange(image_height))  # Shape: (H, W)
    
    # 2. Converti in coordinate omogenee dei pixel
    pixel_coords = np.stack([u, v, np.ones_like(u)], axis=-1)  # Shape: (H, W, 3)
    
    # 3. Calcola d_local = K^{-1} * pixel_coords
    K_inv = np.linalg.inv(intrinsics)
    d_camera = np.einsum('ij,hwj->hwi', K_inv, pixel_coords)  # Shape: (H, W, 3)
    
    # 4. Calcola la distanza euclidea di ciascun pixel dal centro della camera
    distances = np.linalg.norm(d_camera, axis=-1)  # Shape: (H, W)

    return distances
 
# ************************** PATHS **************************
cameraTxt_path = '../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/sparse/images.txt'
imgs_folder = "../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/images"
depth_map_folder = '../datasets/colmap_reconstructions/water_bottle_gui_pinhole_1camera/stereo/depth_maps'

cameraTxt_path = '../datasets/colmap_reconstructions/colmap_output_simple_radial/sparse/cameras.txt'
imagesTxt_path = '../datasets/colmap_reconstructions/colmap_output_simple_radial/sparse/images.txt'
imgs_folder = "../datasets/colmap_reconstructions/colmap_output_simple_radial/dense/images"
depth_map_folder = '../datasets/colmap_reconstructions/colmap_output_simple_radial/dense/stereo/depth_maps'

cameraTxt_path = '../datasets/colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../datasets/colmap_reconstructions/cavignal-bench_pinhole_1camera/sparse/images.txt'
imgs_folder = "../datasets/colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/images"
depth_map_folder = '../datasets/colmap_reconstructions/cavignal-bench_pinhole_1camera/dense/stereo/depth_maps'


cameraTxt_path = '../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/sparse/cameras.txt'
imagesTxt_path = '../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/sparse/images.txt'
imgs_folder = "../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/dense/images"
depth_map_folder = '../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/dense/stereo/depth_maps'
depth_map_fitted_folder = '../datasets/colmap_reconstructions/cavignal-fountain_pinhole_1camera/depth_after_fitting/exp_fit_da_non_metric_and_colmap_true_points'


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

            texture_width = int(single_camera_info[2])
            texture_height = int(single_camera_info[3])

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
print(" Width: ", texture_width)
print(" Height: ", texture_height)
print(" fx: ", fx)
print(" fy: ", fy)
print(" cx: ", cx)
print(" cy: ", cy)  

if 'k1' in locals():
    print(" k1: ", k1)
if 'k2' in locals():
    print(" k2: ", k2)

#intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy) #alternatively
intrinsics = o3d.camera.PinholeCameraIntrinsic(texture_width, texture_height, intrinsic_matrix)

# ************************** EXTRACT EXTRINSICS FROM IMAGES.TXT FILE **************************
# Extrinsic matrix:
# [r1.1, r1.2, r1.3, tx]
# [r2.1, r2.2, r2.3, ty]
# [r3.1, r3.2, r3.3, tz]
# [0,    0,    0,    1 ]

# LINESET to draw camera directions in 3d as 'vectors'
lineset = o3d.geometry.LineSet()
all_points = []
all_lines = []
cameras_coords = []

# COMMON
texture_width, texture_height = 2048, 1024

# FORWARD MAPPING - comment if using inverse map.
'''
texture = defaultdict(list)
'''

# INVERSE MAPPING - comment if using forward map
texture = np.zeros((texture_height, texture_width, 3), dtype=np.uint8)
z_buffer = np.full((texture_height, texture_width), np.inf)
z_buffer_inverse = np.full((texture_height, texture_width), -np.inf)

# LOOP INFO
count = 0
count_imgs = 0
cameras_info = []
cameras_extrinsics = []
# Read 1 image every 'skip'
# e.g. If I have 10 imgs and skip = 3, read images:
# 3, 6, 9
skip = 1 # if 1: do not skip imgs
print("-- You are reading 1 image every ", skip)

with open(imagesTxt_path, 'r') as f:
    for line in f:    
        # Ignore comments
        if not line.startswith("#"):
            count+=1

            if(count > 0):
                if count % 2 != 0: # Read every other line (skip the second line for every image)
                    count_imgs += 2
    
                    if count_imgs % skip == 0: # salta tot righe
                        
                        single_camera_info = line.split() # split every field in line
                        cameras_info.append(single_camera_info) # and store them as separate fields as list in a list ( [ [] ] )

                        print("--- Img num ", (count_imgs/skip)/2 if skip %2 != 0 else count_imgs/skip)
                        print("-- Img filename ", single_camera_info[9])
                        print("Count: ", count)

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

                        # ------------ FIND CAMERA DIRECTION AND PREPARE TO DRAW IT AS A VECTOR IN 3D SPACE 

                        # Extract camera direction vector (forward vector)
                        # rotation_matrix = extrinsics_matrix[:3, :3]  # I already have the rot matrix, keep it commented
                        forward_vector = -rotation_matrix[:, 2]
                        
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

                        # ------------ READ IMAGES

                        # Take the image file name
                        img_filename = single_camera_info[9]

                        # Read the image
                        img_path = os.path.join(imgs_folder, img_filename)
                        img = np.asarray(Image.open(img_path))   

                        
                        # ----- IMAGE TEXTURE FORWARD MAPPING
                        # Comment block if using inverse mapping
                        '''
                        #(image, intrinsics, extrinsics)
                        texel_coords, colors = forward_mapping_project_pixels(img, intrinsic_matrix, extrinsics_matrix, width, height)

                        # Add colors and metadata to texels
                        # do not include depth
                        for (u_texel, v_texel), color in zip(texel_coords, colors):
                            texture[(u_texel, v_texel)].append({
                                "color": color,
                                "img_id": single_camera_info[0]
                            })
                        '''

                        # ----- IMAGE TEXTURE INVERSE MAPPING
                        # Comment block if using forward mapping

                        # con zbuffer
                        image_width = img.shape[1]
                        image_height = img.shape[0]

                        '''
                        --- METODI PER Z BUFFER INVERSE MAPPING
                        - COMMENTA UN BLOCCO SE SI USA UN METODO DIVERSO                        
                        '''

                        '''
                        # METODO 1 - DISTANZA DA CENTRO DELLA CAMERA
                        # trova distanza di ciascun pixel dal centro della camera
                        distances = calculate_pixel_distances_from_camera_center(intrinsic_matrix, image_width, image_height)
                        
                        colors, valid_mask, mapped_distances = inverse_mapping(
                                                                texture_width, texture_height,
                                                                extrinsics_matrix, intrinsic_matrix, img,
                                                                image_width, image_height, distances
                                                            )
                        
                        
                        # Aggiorna la texture usando il buffer delle distanze (z-buffer inverso)
                        farther_mask = valid_mask & (mapped_distances > z_buffer_inverse)
                        z_buffer_inverse[farther_mask] = mapped_distances[farther_mask]
                        texture[farther_mask] = colors[farther_mask]
                        '''

                        '''
                        # METODO 2 - DISTANZA USANDO DEPTH MAP
                        '''
                        depth_map_filename = img_filename + "_depth.npy" 
                        depth_map_path = os.path.join(depth_map_fitted_folder, depth_map_filename)

                        depth_map = np.load(depth_map_path)
                        
                        colors, valid_mask, depth_texture = inverse_mapping(
                                                            texture_width, texture_height,
                                                            extrinsics_matrix, intrinsic_matrix, img,
                                                            image_width, image_height, depth_map
                                                            )
                        
                        

                        ''''                        # - alternativa: tieni pixel + lontani
                        # Aggiorna la texture usando il buffer delle distanze (z-buffer inverso)
                        # tieni solo pixel più lontani - aggiorna texture solo se nuovo valore di profondità è maggiore
                        farther_mask = valid_mask & (depth_texture > z_buffer_inverse) # scelgo punti più lontani
                        z_buffer_inverse[farther_mask] = depth_texture[farther_mask]
                        texture[farther_mask] = colors[farther_mask]
                        '''
                        
                        # - alternativa a tenere i pixel + lontani:
                        # usa una soglia di distanza
                        # Applica la soglia: considera solo i punti con profondità maggiore di depth_threshold
                        depth_threshold = 20
                        above_threshold_mask = valid_mask & (depth_texture > depth_threshold)
                        # Scegli i pixel più lontani tra quelli che superano la soglia
                        farther_mask = above_threshold_mask & (depth_texture > z_buffer_inverse)                        
                        # Aggiorna solo i texel validi
                        z_buffer_inverse[farther_mask] = depth_texture[farther_mask]
                        texture[farther_mask] = colors[farther_mask]                        

                        '''
                        # - alternativa: tieni pixel + vicini
                        # tieni pixel + vicini z buffer classico
                        closer_mask = valid_mask & (depth_texture < z_buffer)  # Scegli i punti più vicini
                        z_buffer[closer_mask] = depth_texture[closer_mask]  # Aggiorna la profondità minima
                        texture[closer_mask] = colors[closer_mask]  # Aggiorna il colore
                        '''

                        '''
                        TEST CON BLENDING - INVERSE MAPPING
                        '''

                        '''
                        # elabora l'immagine corrente
                        colors, valid_mask = inverse_mapping(
                            width, height,
                            extrinsics_matrix, intrinsic_matrix, img,
                            image_width = img.shape[1], image_height = img.shape[0]
                        )

                        # aggiorna la texture finale con il blending
                        texture[valid_mask] += colors[valid_mask]
                        '''
                        
plt.imshow(z_buffer_inverse, cmap='viridis')
plt.colorbar()
plt.title("Buffer delle Distanze (Inverso)")
plt.show()

# ********* Post-processing: create texture

# FORWARD MAPPING - comment if using inverse map.
'''
equirectangular_image = np.zeros((height, width, 3), dtype=np.uint8)
# BLENDING DI OGNI TEXEL - risultato un po' schifo
for (u_texel, v_texel), pixel_stack in texture.items():
    # Combina i colori (tipo media)
    colors = np.array([p["color"] for p in pixel_stack])
    equirectangular_image[v_texel, u_texel] = np.mean(colors, axis=0).astype(np.uint8)
'''

# INVERSE MAPPING - comment if using forward map.
equirectangular_image = texture

#zbuffer: non fare nulla va bne così

'''
# blending
# Normalizza i colori: media ponderata
texture = np.clip(texture, 0, 255).astype(np.uint8)
'''

# View the image
plt.imshow(equirectangular_image)
plt.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()


# ------- SHOW CAMERAS IN 3D (RED) + FORWARD VECTOR (GREEN)
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