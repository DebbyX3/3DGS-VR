import open3d as o3d
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('TkAgg')
import pylab as plt
import math
from collections import defaultdict
import cv2

def feature_matching(img1, img2):
    # Convertiamo le immagini in scala di grigi
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Inizializziamo l'ORB detector
    orb = cv2.ORB_create()

    # Troviamo i keypoints e i descrittori per entrambe le immagini
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Creiamo un matcher di descrittori
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Troviamo le corrispondenze
    matches = bf.match(des1, des2)

    # Ordiniamo le corrispondenze in base alla distanza
    matches = sorted(matches, key = lambda x:x.distance)

    # Restituiamo le corrispondenze trovate
    return kp1, kp2, matches


# Funzione per allineare le immagini
def align_images(img1, img2, kp1, kp2, matches):
    # Estraiamo le coordinate dei punti di corrispondenza
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calcoliamo l'omografia (matrice di trasformazione)
    M, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Applichiamo l'omografia all'immagine 1 per allinearla con l'immagine 2
    aligned_img = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))

    return aligned_img, M

# Funzione di blending basato sulla depth map
def depth_guided_blending(img1, img2, depth_map1, depth_map2):
    # Crea una maschera basata sulla depth map (più distanza = più peso)
    mask1 = depth_map1 > depth_map2
    mask2 = depth_map2 > depth_map1

    # Combinazione lineare usando la maschera
    blended_img = np.zeros_like(img1)
    blended_img[mask1] = img1[mask1]
    blended_img[mask2] = img2[mask2]

    return blended_img

# Funzione di proiezione delle immagini su mappa 360°
def project_to_360(img, intrinsic, extrinsic):
    # La proiezione della tua immagine sulla texture 360° equirettangolare
    # qui andrà il codice che hai sviluppato per il warping delle immagini.
    # Ad esempio, si applica il warping utilizzando le matrici intrinseche ed estrinseche.
    # Il risultato finale dovrebbe essere l'immagine proiettata sulla mappa equirettangolare.
    return img  # Placeholder: sostituisci con il codice di proiezione esistente

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
def inverse_mapping(texture_width, texture_height, extrinsics, intrinsics, image, image_width, image_height, pixel_distances, M=None):   
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

    # Se abbiamo una matrice M di trasformazione, la applichiamo
    if M is not None:
        ones = np.ones_like(u_image[valid_mask])
        points = np.stack([u_image[valid_mask], v_image[valid_mask], ones], axis=-1)
        transformed_points = np.einsum('ij, nj -> ni', M, points.reshape(-1, 3))

        # Normalizza e aggiorna le coordinate
        u_image[valid_mask] = (transformed_points[:, 0] / transformed_points[:, 2]).astype(int)
        v_image[valid_mask] = (transformed_points[:, 1] / transformed_points[:, 2]).astype(int)

    # escludi punti fuori dai bordi - no clipping
    valid_mask &= (u_image >= 0) & (u_image < image_width)
    valid_mask &= (v_image >= 0) & (v_image < image_height)


    ''''''
    # se attivo zbuffer

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
#intrinsics = o3d.camera.PinholeCameraIntrinsic(texture_width, texture_height, intrinsic_matrix)

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
cameras_extrinsics = {}
images_info = {}
depth_info = {}
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
                        
                        # Take the image file name
                        img_filename = single_camera_info[9]

                        # Take image id
                        img_id = int(single_camera_info[0])

                        # CREATE ROTATION MATRIX 'R' FROM QUATERNIONS
                        quaternions = np.array([single_camera_info[1], single_camera_info[2], single_camera_info[3], single_camera_info[4]]) # numpy array
                        rotation_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternions)

                        # CREATE TRANSLATION VECTOR T
                        translation = np.array([single_camera_info[5], single_camera_info[6], single_camera_info[7]], dtype = float)

                        # CREATE EXTRINSICS MATRIX                
                        extrinsics_matrix = np.vstack([np.hstack([rotation_matrix, translation.reshape(3, 1)]), 
                                                        np.array([0, 0, 0, 1])])

                        cameras_extrinsics[img_id] = {'filename': img_filename, 'extrinsics': extrinsics_matrix}

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

                        # Read the image
                        img_path = os.path.join(imgs_folder, img_filename)
                        img = np.asarray(Image.open(img_path))   

                        # Aggiungi il nome del file e l'immagine alla lista
                        images_info[img_id] = {'filename': img_filename, 'image': img}

                        # ------------ READ DEPTH MAPS
                        depth_map_filename = img_filename + "_depth.npy" 
                        depth_map_path = os.path.join(depth_map_fitted_folder, depth_map_filename)
                        depth_map = np.load(depth_map_path)

                        # Aggiungi il nome del file e l'immagine alla lista
                        depth_info[img_id] = {'filename': depth_map_filename, 'depth': depth_map}

                        
# cicla sulle immagini salvate
# Funzione principale per la generazione della texture 360°
final_texture = np.zeros((texture_height, texture_width, 3), dtype=np.uint8)
depth_buffer = np.full((texture_height, texture_width), -np.inf)

for i in range(2, len(images_info)):
    img1 = images_info[i-1]['image']
    img2 = images_info[i]['image']
    depth_map1 = depth_info[i-1]['depth']
    depth_map2 = depth_info[i]['depth']
    
    # Step 1: Feature matching
    kp1, kp2, matches = feature_matching(img1, img2)
    
    # Step 2: Allineamento delle immagini
    aligned_img, M = align_images(img1, img2, kp1, kp2, matches)
    
    # Step 3: Blending delle immagini usando le depth maps
    blended_img = depth_guided_blending(aligned_img, img2, depth_map1, depth_map2)

    # Step 3.5: Trasforma direttamente l'immagine con M PRIMA della proiezione
    height, width = blended_img.shape[:2]
    stitched_img = cv2.warpPerspective(blended_img, M, (width, height))
    stitched_depth = cv2.warpPerspective(depth_map2, M, (width, height))

    # Step 4: Proiezione delle immagini sulla texture 360°
    projected_colors, valid_mask, projected_depths = inverse_mapping(
        texture_width, texture_height, cameras_extrinsics[i]['extrinsics'], intrinsic_matrix, 
        stitched_img, stitched_img.shape[1], stitched_img.shape[0], stitched_depth)
    
    # Step 5: Combina la proiezione nella texture finale
    update_mask = valid_mask & (projected_depths > depth_buffer)
    final_texture[update_mask] = projected_colors[update_mask]
    depth_buffer[update_mask] = projected_depths[update_mask]

    print ("Done img ", i)

# View the image
plt.imshow(final_texture)
plt.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
plt.show()








""" Stitching + Proiezione su texture 360° """
'''
final_texture = np.zeros((texture_height, texture_width, 3), dtype=np.uint8)
depth_buffer = np.full((texture_height, texture_width), -np.inf)

prev_img = None
prev_depth = None
prev_extrinsic = None

for img_id, extrinsics in zip(sorted(images_info.keys()), cameras_extrinsics):
    image = images_info[img_id]['image']
    depth_map = depth_info[img_id]

    if prev_img is not None:
        # Step 1: Stitching tra immagini adiacenti
        kp1, kp2, matches = feature_matching(prev_img, image)
        aligned_img, _ = align_images(prev_img, image, kp1, kp2, matches)
        blended_img = depth_guided_blending(aligned_img, image, prev_depth, depth_map)
    else:
        blended_img = image  # Prima immagine senza blending

    # Step 2: Proiezione con inverse mapping
    projected_colors, valid_mask, projected_depths = inverse_mapping(
        texture_width, texture_height, extrinsics, intrinsics, blended_img, 
        blended_img.shape[1], blended_img.shape[0], depth_map
    )

    # Step 3: Inserimento nella texture 360° con depth buffer
    update_mask = valid_mask & (projected_depths > depth_buffer)
    final_texture[update_mask] = projected_colors[update_mask]
    depth_buffer[update_mask] = projected_depths[update_mask]

    prev_img = image
    prev_depth = depth_map
    prev_extrinsic = extrinsics
'''