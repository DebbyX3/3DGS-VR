import numpy as np
import cv2
import pycolmap
import argparse
import collections
import os
import struct
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import open3d as o3d

#### FROM COLMAP CODEBASE ####

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"]
)
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

class Image(BaseImage):
    def __new__(cls, id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self = super(Image, cls).__new__(cls, id, qvec, tvec, camera_id, name, xys, point3D_ids)
        self._name = name
        return self

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)

def read_images_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q"
            )[0]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")

def read_cameras_binary(path_to_model_file):
    """
    see: src/colmap/scene/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras

#### END FROM COLMAP CODEBASE ####

def parse_cameras(cameras_bin_path):
    # ************** READ COLMAP CAMERA FILE    

    # ***  WARNING: THIS SCRIPT ASSUMES THAT ALL CAMERAS HAVE THE SAME INTRINSICS ***
    # ***  SO IN THE CAMERA FILE WE WILL ONLY READ THE FIRST CAMERA INTRINSICS ***
    # *** (ALSO BEACUSE THERE IS ONLY ONE CAMERA IN THE CAMERA FILE IF THEY SHARE THE SAME INTRINSICS) ***

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

    cameras_bin_info = read_cameras_binary(cameras_bin_path) 

    cameras_info = {}

    for cam_id, cam in cameras_bin_info.items():
        # Camera info contains:
        # CAMERA_ID  MODEL   WIDTH   HEIGHT  PARAMS[]
        # 0          1       2       3       4   5   6   7   8
        # Where PARAMS[] are:
        # SIMPLE_PINHOLE: fx (fx = fy), cx, cy      1 focal length and principal point
        # PINHOLE: fx, fy, cx, cy                   2 focal lenghts and principal point
        # SIMPLE_RADIAL: fx (fx = fy), cx, cy, k1   1 focal length, principal point and radial distortion
        # RADIAL: fx (fx = fy), cx, cy, k1, k2      1 focal lengths, principal point and 2 radial distortions

        if cam.model == "PINHOLE":
            cameras_info[cam_id] = {'id': cam_id, 
                                    'type': cam.model, 
                                    'width': cam.width, 
                                    'height': cam.height, 
                                    'fx': cam.params[0], 
                                    'fy': cam.params[1], 
                                    'cx': cam.params[2], 
                                    'cy': cam.params[3]}
        

        print("--- Camera ID: " + str(cameras_info[cam_id]['id']) + " - " + cameras_info[cam_id]['type'])
        print(" Width: ", cameras_info[cam_id]['width'])
        print(" Height: ", cameras_info[cam_id]['height'])
        print(" fx: ", cameras_info[cam_id]['fx'])
        print(" fy: ", cameras_info[cam_id]['fy'])
        print(" cx: ", cameras_info[cam_id]['cx'])
        print(" cy: ", cameras_info[cam_id]['cy'])  

        '''if 'k1' in locals():
            print(" k1: ", k1)
        if 'k2' in locals():
            print(" k2: ", k2)'''

        break    # We only need the first camera intrinsics (assume all cameras have the same intrinsics)  

    # Create the camera intrinsic matrix with the first (and only one) camera's intrinsics
    intrinsic_matrix = np.array([[cameras_info[1]['fx'],    0,                      cameras_info[1]['cx']],
                                [0,                         cameras_info[1]['fy'],  cameras_info[1]['cy']],
                                [0,                         0,                      1]])

    return cameras_info, intrinsic_matrix

def parse_images(images_bin_path):
    """Parsa il file images.txt per estrarre le pose delle immagini"""
    # Images info contains two rows per image:

    #   1st row has:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   0         1   2   3   4   5   6   7   8          9

    #   2nd row has:
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    #   Example of this row:
    #   2362.39 248.498 58396       1784.7 268.254 59027        1784.7 268.254 -1
    #   X       Y       POINT3D_ID  X      Y       POINT3D_ID   X      Y       POINT3D_ID
    #   the last keypoint does not observe a 3D point in the reconstruction as the 3D point identifier is -1

    images_bin_info = read_images_binary(images_bin_path) 

    images_info = {}

    for img_id, img in images_bin_info.items():
        # COLMAP: ROTATION MATRIX 'R' FROM QUATERNION - FROM WORLD TO CAMERA
        qw, qx, qy, qz = img.qvec # please note that COLMAP writes/uses the quaternion in the order [qw, qx, qy, qz]
        rot_from_w_to_c = R.from_quat([qx, qy, qz, qw]).as_matrix() # but scipy uses the order [qx, qy, qz, qw] for quaternions

        print("Rotation matrix world to cam:\n", rot_from_w_to_c)
        print("rotation matrix cam to world (converted)\n", rot_from_w_to_c.transpose())

        # COLMAP: TRANSLATION VECTOR 'T' - FROM WORLD TO CAMERA
        tx, ty, tz = img.tvec
        trans_from_w_to_c = np.array([tx, ty, tz], dtype = float).reshape(3, 1)

        print("\nTranslation vector world to camera:\n", trans_from_w_to_c)
        print("Translation vector camera to world (converted)\n", -(rot_from_w_to_c.transpose()) @ trans_from_w_to_c.reshape(3, 1))


        images_info[img_id] = {'id': img_id, 
                                'filename': img.name,
                                'rot_from_w_to_c': rot_from_w_to_c,
                                'trans_from_w_to_c': trans_from_w_to_c,
                                'camera_id': img.camera_id,
                                'points_2d': img.xys,
                                'point3D_ids': img.point3D_ids
                                }
    
        print("--- Image ID: " + str(images_info[img_id]['id']) + " - " + images_info[img_id]['filename'])
        
    return images_info

def camera_depth_to_scene_depth(depth_img, intrinsics_matrix_K, rot_from_w_to_c, trans_from_w_to_c, scene_center_world_coords, depth_scale=1.0):
    # NOTA: QUA MANTENGO 0 DOVE LA DEPTH ERA 0 O MINORE DI 0, 
    # MA RICOSTRUISCO LA DEPTH COME DISTANZA DAL CENTRO DELLA SCENA

    height_depth, width_depth = depth_img.shape

    # Inverti matrice intrinseci    
    intrinsics_K_inv = np.linalg.inv(intrinsics_matrix_K)

    # Convert quaternion from world-to-camera (colmap) to camera-to-world
    # Please remember the convention! R_wc means "rotation from camera to world"! and NOT viceversa!
    rot_from_c_to_w = rot_from_w_to_c.transpose()  # camera-to-world: R_wc =  R_cw.T

    # Convert translation from world-to-camera (colmap) to camera-to-world
    trans_from_c_to_w = -rot_from_c_to_w @ trans_from_w_to_c # camera-to-world: t_wc = -R_wc @ t_cw (oppure: t_wc = -R_cw.T @ t_cw)

    # Prepara la mappa delle distanze
    scene_depth = np.zeros((height_depth, width_depth), dtype=np.float32)

    # -------------
    
    # Crea griglia pixel (u,v)
    u_coords, v_coords = np.meshgrid(np.arange(width_depth), np.arange(height_depth))

    # Costruisci array pixel omogenei. shape (3, H*W)
    pixels_homog = np.stack([u_coords.ravel(), v_coords.ravel(), np.ones_like(u_coords).ravel()])

    # Prendi depth e applica scale. shape (H*W)
    depth_values = depth_img.ravel().astype(np.float32) * depth_scale

    #butta i val a 255 (max) a un valore altissimo per prova
    #depth_values[np.argwhere(depth_values == np.amax(depth_values))] = 255*1.2

    # Scarta depth = 0 per evitare punti invalidi
    valid_mask = depth_values > 0
    depth_valid = depth_values[valid_mask]
    pixels_valid = pixels_homog[:, valid_mask]

    # calculate 3D points in camera space. shape (3, N)
    # calculate the ray from the camera center through the pixel (u,v) trasforming it in coord in image plane. 
    # Then, scale it by the depth value, obtaining the 3d position in the camera space

    #original
    points_camera = intrinsics_K_inv @ pixels_valid 
    points_camera *= depth_valid  # scala per la profondità # ricostruzione punto 3D in spazio camera

    #new
    #points_camera = intrinsics_K_inv @ pixels_valid 
    #points_camera = pixels_valid * depth_valid  # scala per la profondità # ricostruzione punto 3D in spazio camera

    # Trasforma in coordinate mondo. shape (3, N)
    # Trasforma il punto 3D da spazio camera a spazio mondo
    points_world = rot_from_c_to_w @ points_camera + trans_from_c_to_w.reshape(3,1) # trasformazione in mondo

    # calculate new depth from the center of the scene: euclidean distance
    diffs = points_world.T - scene_center_world_coords.reshape(1,3)
    dists = np.linalg.norm(diffs, axis=1)

    # Ricostruisci mappa depth finale con zeri dove depth era zero
    scene_depth = np.zeros_like(depth_values, dtype=np.float32)
    scene_depth[valid_mask] = dists
    scene_depth = scene_depth.reshape((height_depth, width_depth))
    

    '''
    # QUESTO VA BENISSIMO MA è MOLTO LENTO
    
    # Cicla su ogni pixel
    for v in range(height_depth):
        for u in range(width_depth):
            depth_value = float(depth_img[v, u]) * depth_scale
            if depth_value == 0:
                continue  # niente informazione

            pixel = np.array([u, v, 1.0])

            # calculate the 3D point in camera space
            # calculate the ray from the camera center through the pixel (u,v) trasforming it in coord in image plane. 
            # Then, scale it by the depth value, obtaining the 3d position in the camera space
            point_in_camera_space = depth_value * (intrinsics_K_inv @ pixel)  # ricostruzione punto 3D in spazio camera

            # Trasforma il punto 3D da spazio camera a spazio mondo
            point_in_world_space = rot_from_c_to_w @ point_in_camera_space + trans_from_c_to_w.flatten()  # trasformazione in mondo

            # calculate new depth from the center of the scene - euclidean distance
            dist = np.linalg.norm(point_in_world_space - scene_center_world_coords)
            scene_depth[v, u] = dist
    '''
    return scene_depth

def camera_depth_to_scene_depth_corrected(depth_img, intrinsics_matrix_K, rot_from_w_to_c, trans_from_w_to_c, scene_center_world_coords):
    """
    Trasforma depth map da camera a distanze dal centro scena.
    IMPORTANTE: Questa versione non usa i valori di depth come distanze metriche,
    ma li usa solo per ordinare i punti lungo ciascun raggio.
    """
    height_depth, width_depth = depth_img.shape

    # Inverti matrice intrinseci    
    intrinsics_K_inv = np.linalg.inv(intrinsics_matrix_K)

    # Convert da world-to-camera a camera-to-world
    rot_from_c_to_w = rot_from_w_to_c.transpose()
    trans_from_c_to_w = -rot_from_c_to_w @ trans_from_w_to_c

    # Mappa delle distanze dal centro scena
    scene_depth = np.zeros((height_depth, width_depth), dtype=np.float32)

    # Crea griglia pixel (u,v)
    u_coords, v_coords = np.meshgrid(np.arange(width_depth), np.arange(height_depth))
    pixels_homog = np.stack([u_coords.ravel(), v_coords.ravel(), np.ones_like(u_coords).ravel()])

    # Valori depth normalizzati (0-1) invece di usarli come metrici
    depth_values = depth_img.ravel().astype(np.float32) / 255.0

    # Maschera per pixel validi (depth > 0)
    valid_mask = depth_values > 0
    depth_valid = depth_values[valid_mask]
    pixels_valid = pixels_homog[:, valid_mask]

    # CHIAVE: Calcola le DIREZIONI dei raggi dalla camera, non i punti 3D
    # Questi sono vettori unitari che indicano la direzione di ciascun pixel
    ray_directions_camera = intrinsics_K_inv @ pixels_valid
    #ray_directions_camera = ray_directions_camera / np.linalg.norm(ray_directions_camera, axis=0)  # Normalizza

    # Trasforma le direzioni in coordinate mondo
    ray_directions_world = rot_from_c_to_w @ ray_directions_camera

    # Posizione della camera in coordinate mondo
    camera_position_world = trans_from_c_to_w.flatten()

    # Per ogni raggio, calcola dove interseca una sfera ideale centrata nel centro della scena
    # La "profondità relativa" della depth map determina quanto lontano lungo il raggio
    
    # Distanza base dalla camera al centro della scena
    camera_to_center = np.linalg.norm(camera_position_world - scene_center_world_coords)
    
    # Usa i valori di depth (0-1) per interpolare lungo ciascun raggio
    # depth_valid = 0 significa vicino alla camera, depth_valid = 1 significa lontano
    # Puoi regolare questi fattori in base al tuo dataset
    min_distance_factor = 0.1  # frazione della distanza camera-centro per depth=0
    max_distance_factor = 100.0  # multiplo della distanza camera-centro per depth=1
    
    min_distance = camera_to_center * min_distance_factor
    max_distance = camera_to_center * max_distance_factor
    
    # Interpola la distanza lungo ciascun raggio basandosi sui valori di depth
    ray_distances = min_distance + depth_valid * (max_distance - min_distance)
    
    # Calcola i punti 3D lungo i raggi
    points_world = camera_position_world.reshape(3, 1) + ray_directions_world * ray_distances
    
    # Calcola le distanze dal centro della scena
    diffs = points_world.T - scene_center_world_coords.reshape(1, 3)
    distances_from_center = np.linalg.norm(diffs, axis=1)

    # Ricostruisci la mappa finale
    scene_depth_flat = np.zeros_like(depth_values, dtype=np.float32)
    scene_depth_flat[valid_mask] = distances_from_center
    scene_depth = scene_depth_flat.reshape((height_depth, width_depth))

    return scene_depth

def camera_depth_to_scene_depth_simple(depth_img, intrinsics_matrix_K, rot_from_w_to_c, trans_from_w_to_c, scene_center_world_coords):
    """
    Trasforma depth map da camera a distanze dal centro scena.
    Mantiene la coerenza relativa dei valori originali.
    """
    height_depth, width_depth = depth_img.shape

    # Inverti matrice intrinseci    
    intrinsics_K_inv = np.linalg.inv(intrinsics_matrix_K)

    # Convert da world-to-camera a camera-to-world
    rot_from_c_to_w = rot_from_w_to_c.transpose()
    trans_from_c_to_w = -rot_from_c_to_w @ trans_from_w_to_c
    camera_position_world = trans_from_c_to_w.flatten()

    # Mappa delle distanze dal centro scena
    scene_depth = np.zeros((height_depth, width_depth), dtype=np.float32)

    # Crea griglia pixel (u,v)
    u_coords, v_coords = np.meshgrid(np.arange(width_depth), np.arange(height_depth))
    pixels_homog = np.stack([u_coords.ravel(), v_coords.ravel(), np.ones_like(u_coords).ravel()])

    # Usa i valori depth così come sono (non normalizzarli!)
    depth_values = depth_img.ravel().astype(np.float32)

    # Maschera per pixel validi
    valid_mask = depth_values > 0
    depth_valid = depth_values[valid_mask]
    pixels_valid = pixels_homog[:, valid_mask]

    # Calcola le direzioni dei raggi (vettori unitari)
    ray_directions_camera = intrinsics_K_inv @ pixels_valid
    ray_directions_camera = ray_directions_camera / np.linalg.norm(ray_directions_camera, axis=0)

    # Trasforma le direzioni in coordinate mondo
    ray_directions_world = rot_from_c_to_w @ ray_directions_camera

    # CHIAVE: Usa i valori di depth per posizionare i punti lungo i raggi
    # I valori 0-255 diventano "unità di profondità" lungo ciascun raggio
    points_world = camera_position_world.reshape(3, 1) + ray_directions_world * depth_valid

    # Calcola le distanze dal centro della scena
    diffs = points_world.T - scene_center_world_coords.reshape(1, 3)
    distances_from_center = np.linalg.norm(diffs, axis=1)

    # Ricostruisci la mappa finale
    scene_depth_flat = np.zeros_like(depth_values, dtype=np.float32)
    scene_depth_flat[valid_mask] = distances_from_center
    scene_depth = scene_depth_flat.reshape((height_depth, width_depth))

    return scene_depth

def camera_depth_to_scene_depth_practical(depth_img, intrinsics_matrix_K, rot_from_w_to_c, trans_from_w_to_c, scene_center_world_coords, depth_scale_factor=0.01):
    """
    Trasforma depth map da camera a "proxy" di distanze dal centro scena.
    Usa un fattore di scala arbitrario ma consistente per dare significato ai valori 0-255.
    
    depth_scale_factor: converti 0-255 in unità di scena (es. 0.01 significa 255 → 2.55 unità)
    """
    height_depth, width_depth = depth_img.shape

    # Inverti matrice intrinseci    
    intrinsics_K_inv = np.linalg.inv(intrinsics_matrix_K)

    # Convert da world-to-camera a camera-to-world
    rot_from_c_to_w = rot_from_w_to_c.transpose()
    trans_from_c_to_w = -rot_from_c_to_w @ trans_from_w_to_c
    camera_position_world = trans_from_c_to_w.flatten()

    # Mappa delle distanze dal centro scena
    scene_depth = np.zeros((height_depth, width_depth), dtype=np.float32)

    # Crea griglia pixel (u,v)
    u_coords, v_coords = np.meshgrid(np.arange(width_depth), np.arange(height_depth))
    pixels_homog = np.stack([u_coords.ravel(), v_coords.ravel(), np.ones_like(u_coords).ravel()])

    # CHIAVE: Converti 0-255 in "unità di scena" usando un fattore consistente
    depth_values = depth_img.ravel().astype(np.float32) * depth_scale_factor

    # Maschera per pixel validi
    valid_mask = depth_values > 0
    depth_valid = depth_values[valid_mask]
    pixels_valid = pixels_homog[:, valid_mask]

    # Calcola le direzioni dei raggi (vettori unitari)
    ray_directions_camera = intrinsics_K_inv @ pixels_valid
    ray_directions_camera = ray_directions_camera / np.linalg.norm(ray_directions_camera, axis=0)

    # Trasforma le direzioni in coordinate mondo
    ray_directions_world = rot_from_c_to_w @ ray_directions_camera

    # Posiziona i punti lungo i raggi usando le depth scalate
    points_world = camera_position_world.reshape(3, 1) + ray_directions_world * depth_valid

    # Calcola le distanze dal centro della scena
    diffs = points_world.T - scene_center_world_coords.reshape(1, 3)
    distances_from_center = np.linalg.norm(diffs, axis=1)

    # Ricostruisci la mappa finale
    scene_depth_flat = np.zeros_like(depth_values, dtype=np.float32)
    scene_depth_flat[valid_mask] = distances_from_center
    scene_depth = scene_depth_flat.reshape((height_depth, width_depth))

    return scene_depth

# ALTERNATIVA: Se le depth map hanno già una scala ragionevole per la tua scena
def camera_depth_to_scene_depth_auto_scale(depth_img, intrinsics_matrix_K, rot_from_w_to_c, trans_from_w_to_c, scene_center_world_coords):
    """
    Versione che stima automaticamente una scala ragionevole basata sulla distanza camera-centro.
    """
    # Calcola posizione camera
    rot_from_c_to_w = rot_from_w_to_c.transpose()
    trans_from_c_to_w = -rot_from_c_to_w @ trans_from_w_to_c
    camera_position_world = trans_from_c_to_w.flatten()
    
    # Distanza camera-centro come riferimento
    camera_to_center_dist = np.linalg.norm(camera_position_world - scene_center_world_coords)
    print("Camera to scene center distance:", camera_to_center_dist)
    
    # Scala automatica: 255 dovrebbe corrispondere a circa 2x la distanza camera-centro
    auto_scale = (camera_to_center_dist * 2.0) / 255.0
    print("Auto scale factor for depth:", auto_scale)
    
    return camera_depth_to_scene_depth_practical(depth_img, intrinsics_matrix_K, rot_from_w_to_c, trans_from_w_to_c, scene_center_world_coords, auto_scale)

'''
def depth_to_color_pointcloud(depth_map, rgb_image, intrinsics_matrix_K, sample_ratio=0.1, max_depth=None):
    height, width = depth_map.shape
    fx = intrinsics_matrix_K[0, 0]
    fy = intrinsics_matrix_K[1, 1]
    cx = intrinsics_matrix_K[0, 2]
    cy = intrinsics_matrix_K[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid_mask = depth_map > 0
    if max_depth is not None:
        valid_mask &= (depth_map < max_depth)

    # Estrai indici validi
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth_map[valid_mask]

    # Campiona solo una percentuale dei punti validi
    n_points = len(depth_valid)
    n_sample = int(n_points * sample_ratio)
    if n_sample == 0:
        n_sample = min(1, n_points)  # almeno 1 punto se possibile

    sample_indices = np.random.choice(n_points, size=n_sample, replace=False)

    u_sampled = u_valid[sample_indices]
    v_sampled = v_valid[sample_indices]
    depth_sampled = depth_valid[sample_indices]

    # Calcola coordinate 3D per i punti campionati
    x = (u_sampled - cx) * depth_sampled / fx
    y = (v_sampled - cy) * depth_sampled / fy
    z = depth_sampled
    points = np.stack((x, y, z), axis=-1)

    # Colori corrispondenti
    colors = rgb_image[v_sampled, u_sampled, :].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
'''

cameras_bin_path = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\brg_rm_small_park-FullFrames\\sparse\\cameras.bin"
images_bin_path = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\brg_rm_small_park-FullFrames\\sparse\\images.bin"
images_folder = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\brg_rm_small_park-FullFrames\\images"
#depth_maps_folder = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\brg_rm_small_park-FullFrames\\depth_after_fitting\\exp_fit_da_non_metric_and_colmap_true_points"
#world_output_depth_maps_folder = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\brg_rm_small_park-FullFrames\\depth_after_fitting_and_in_world"
depth_maps_folder = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\depth-anything-estimations\\non-metric_depths\\brg_rm_small_park-FullFrames"
world_output_depth_maps_folder = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\brg_rm_small_park-FullFrames\\depth_anything_maps_in_world"


cameras_bin_path = "../datasets\\colmap_reconstructions\\fields\\sparse\\0\\cameras.bin"
images_bin_path = "../datasets\\colmap_reconstructions\\fields\\sparse\\0\\images.bin"
images_folder = "../datasets\\colmap_reconstructions\\fields\\images"
depth_maps_folder = "../datasets\\colmap_reconstructions\\fields\\video-depth-anything\\1080x1920\\framesInterpolated"
world_output_depth_maps_folder = "../datasets\\colmap_reconstructions\\fields\\video-depth-anything-in-world"


cameras_info, intrinsic_matrix = parse_cameras(cameras_bin_path)
images_info = parse_images(images_bin_path)

# CENTER IN SCENE 'BRG RM SMALL PARK FULL FRAMES' OBTAINED FROM generate_geodesic_points.py
#scene_center_world_coords = np.array([-0.00298899,  0.02030827,  0.08498714]) #borgo roma small
scene_center_world_coords = np.array([-0.07003578, -0.01569284, -0.14668444]) #fields

scene_depths = {}
i = 0

for scene_depth_id, img in images_info.items():
    i += 1

    # IN CASE OF NUMPY DEPTH MAPS
    #depth_filename = img['filename'] + "_depth.npy"
    #original_depth = np.load(depth_maps_folder + "\\" + depth_filename)

    # IN CASE OF PNG DEPTH MAPS
    depth_filename = img['filename']
    original_depth = cv2.imread(depth_maps_folder + "\\" + depth_filename, cv2.IMREAD_GRAYSCALE)

    # IN CASE OF DEPTH MAPS FROM DEPTH ANYTHING
    # Need to invert the map to match the colmap rerpresentation of lower vals = closer point (0: closest, 255: farthest) 
    original_depth = np.invert(original_depth)

    
    scene_depth = camera_depth_to_scene_depth(original_depth, 
                                                intrinsic_matrix,
                                                img['rot_from_w_to_c'],
                                                img['trans_from_w_to_c'],
                                                scene_center_world_coords
                                                )
    
    
    '''
    scene_depth = camera_depth_to_scene_depth_corrected(original_depth, 
                                                intrinsic_matrix,
                                                img['rot_from_w_to_c'],
                                                img['trans_from_w_to_c'],
                                                scene_center_world_coords
                                                )
    '''

    '''
    scene_depth = camera_depth_to_scene_depth_simple(original_depth, 
                                                intrinsic_matrix,
                                                img['rot_from_w_to_c'],
                                                img['trans_from_w_to_c'],
                                                scene_center_world_coords
                                                )
    '''

    '''
    scene_depth = camera_depth_to_scene_depth_auto_scale(original_depth, 
                                                intrinsic_matrix,
                                                img['rot_from_w_to_c'],
                                                img['trans_from_w_to_c'],
                                                scene_center_world_coords
                                                )
    '''

    scene_depths[scene_depth_id] = scene_depth

    # Mostra fianco a fianco le due depth map
    plt.subplot(1, 2, 1)
    plt.title("Depth originale (camera)")
    plt.imshow(original_depth, cmap='inferno')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Depth rispetto al centro scena")
    plt.imshow(scene_depth, cmap='inferno')
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    
    
    # Save new depth map in world coordinates
    #np.save(world_output_depth_maps_folder + "\\" + depth_filename, scene_depth) 
    #print("saved depth map in world coordinates for image ID:", scene_depth_id)


# a quanto pare non posso mostrare la point cloud perchè trasformando le depth in mondo
# mi sono persa la direzione che è fondamentale per ricostrire la point cloud
'''
# Creo la point cloud complessiva vuota
combined_pcd = o3d.geometry.PointCloud()

# Crea una point cloud rossa da tutte le depth map della scena
for scene_depth_id, scene_depth_data in scene_depths.items():
    
    rgb_path = images_folder + '\\' + images_info[scene_depth_id]['filename']
    rgb_image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)  # OpenCV legge BGR, convertiamo in RGB

    pcd = depth_to_color_pointcloud(scene_depth_data, rgb_image, intrinsic_matrix, sample_ratio=0.01)
    combined_pcd += pcd  # accumulo i punti

o3d.visualization.draw_geometries([combined_pcd])
'''