import os
from PIL import Image
import numpy as np

def move_fully_black_images(input_folder, black_folder):
    os.makedirs(black_folder, exist_ok=True)
    moved_files = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            file_path = os.path.join(input_folder, filename)
            try:
                img = Image.open(file_path).convert("RGB")
                img_np = np.array(img)
                if np.all(img_np == 0):
                    target_path = os.path.join(black_folder, filename)
                    os.rename(file_path, target_path)
                    moved_files.append(filename)
            except Exception as e:
                print(f"Errore con {filename}: {e}")
    return moved_files

# Esempio d'uso:
input_folder = "../datasets/colmap_reconstructions/ice_lab_360_2/rect_persp_views-num_20-FOV_90-res_2048x2048"
output_folder = "../datasets/colmap_reconstructions/ice_lab_360_2/rect_persp_views-num_20-FOV_90-res_2048x2048/moved"
moved = move_fully_black_images(input_folder, output_folder)
print(f"Immagini spostate: {moved}")
