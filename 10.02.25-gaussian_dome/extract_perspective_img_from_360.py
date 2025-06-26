import numpy as np
import cv2
import os
from math import pi, tan


# ----- UTILS -----
def rotation_matrix(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def look_rotation(forward, up=np.array([0, 1, 0])):
    f = forward / np.linalg.norm(forward)
    r = np.cross(up, f)
    r /= np.linalg.norm(r)
    u = np.cross(f, r)
    return np.stack((r, u, f), axis=1)

def equirectangular_to_perspective(equi_img, rotation, fov_deg, out_size, intrinsics):
    height, width = out_size
    fov = np.radians(fov_deg)
    fx = fy = (0.5 * width) / tan(fov / 2)
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    i, j = np.meshgrid(np.arange(width), np.arange(height))
    x = (i - cx) / fx
    y = (j - cy) / fy
    z = np.ones_like(x)
    directions = np.stack((x, y, z), axis=-1)
    directions = directions @ rotation.T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    lon = np.arctan2(directions[..., 0], directions[..., 2])
    lat = np.arcsin(-directions[..., 1]) # minus otherwise images are mirrored
    u = (lon / pi + 1.0) * 0.5 * equi_img.shape[1]
    v = (0.5 - lat / pi) * equi_img.shape[0]

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    persp = cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return persp

def mask_bottom(img, band_height=200):
    img = img.copy()
    img[-band_height:, :] = 0
    return img

def adjust_direction(dir, min_y=-0.5, lift_angle_deg=15):
    if dir[1] < min_y:
        forward = dir / np.linalg.norm(dir)
        up = np.array([0, 1, 0])
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        R = rotation_matrix(right, np.radians(lift_angle_deg))
        return R @ forward
    else:
        return dir

def generate_view_directions(num_views):
    directions = []
    ga = pi * (3 - np.sqrt(5))
    for i in range(num_views):
        y = 1 - (i / float(num_views - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = ga * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        directions.append(np.array([x, y, z]))
    return directions

# ----- MAIN PROCESS -----
def process_equirectangular_images(input_folder, output_folder, num_views=15, fov_deg=90, out_res=(2048, 2048), band_height=200):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))])

    for img_name in image_files:
        img_path = os.path.join(input_folder, img_name)
        equi_img = cv2.imread(img_path)
        equi_img = mask_bottom(equi_img, band_height=band_height)

        directions = generate_view_directions(num_views)
        directions = [adjust_direction(d) for d in directions]

        for idx, dir_vec in enumerate(directions):
            rot = look_rotation(dir_vec)
            persp_img = equirectangular_to_perspective(equi_img, rot, fov_deg=fov_deg, out_size=out_res, intrinsics=None)
            out_img_name = f"{os.path.splitext(img_name)[0]}_view{idx:02d}.png"
            out_img_path = os.path.join(output_folder, out_img_name)
            cv2.imwrite(out_img_path, persp_img)

        print("saved frame: ", img_name, " with ", num_views, " views")

# ----- CONFIGURAZIONE -----
EQUIRECT_IMAGE_PATH = '../datasets/colmap_reconstructions/ice_lab_360_2/original_frames/0001.jpg'
INPUT_FOLDER = '../datasets/colmap_reconstructions/ice_lab_360_2/original_frames/'
NUM_VIEWS = 20
FOV_DEGREES = 90
OUTPUT_RES = (2048, 2048)  # (width, height)
OUTPUT_FOLDER = '../datasets/colmap_reconstructions/ice_lab_360_2//' + 'rect_persp_views-num_'+ str(NUM_VIEWS) + '-FOV_' + str(FOV_DEGREES) + '-res_' + str(OUTPUT_RES[0]) + 'x' + str(OUTPUT_RES[1])

process_equirectangular_images(INPUT_FOLDER, OUTPUT_FOLDER, NUM_VIEWS, FOV_DEGREES, OUTPUT_RES, band_height=900)