import numpy as np
import os
import pylab as plt
from PIL import Image
from pathlib import Path

depth_maps_folder = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/video-depth-anything-metric/fromSceneCenter/distances'
images_folder = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/images'
save_images_threshold_folder = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/video-depth-anything-metric/fromSceneCenter/distances_threshold'

i = 0

#show the first 3 distance maps
for subdir, dirs, files in os.walk(images_folder):
    for file in files:

        image_path = os.path.join(subdir, file)
        image_filename_base = Path(file).stem

        depth_filename = image_filename_base + "_distance.npz"
        depth_path = os.path.join(depth_maps_folder, depth_filename)

        if not os.path.exists(depth_path):
            print(f"Distance map not found for image {file}, skipping.")
            continue

        depth_map_file = np.load(depth_path)
        depth_map = depth_map_file['distance']

        plt.imshow(depth_map, cmap='viridis')
        plt.title("Metric Depth Map")
        plt.show()

        i += 1

        if i >= 3:
            break

#ask after showing the first 3 distance maps
# Limit the depth map to a threshold
threshold = float(input("Enter the threshold value for depth: "))
#threshold = 10 #default threshold value


# Loop on all images
for subdir, dirs, files in os.walk(images_folder):
    for file in files:

        image_path = os.path.join(subdir, file)
        image_filename_base = Path(file).stem

        depth_filename = image_filename_base + "_distance.npz"
        depth_path = os.path.join(depth_maps_folder, depth_filename)

        if not os.path.exists(depth_path):
            print(f"Distance map not found for image {file}, skipping.")
            continue

        depth_map_file = np.load(depth_path)
        depth_map = depth_map_file['distance']
        
        # Create a copy of the depth map array
        binary_mask = np.copy(depth_map)

        # ------------------------------

        # *********** FOR BACKGROUND
        # INCLUDE ALL PIXELS WITH DEPTH GREATER OR EQUAL THAN THE THRESHOLD

        # Create a binary mask 
        binary_mask[binary_mask < threshold] = 0.0
        binary_mask[binary_mask >= threshold] = 1.0 # cioè ci va bene anche la threshold stessa

        # -------------------------------
        '''
        # *********** FOR FOREGROUND
        # INCLUDE ALL PIXELS WITH DEPTH LESS THAN THE THRESHOLD

        # Create a binary mask 
        binary_mask[binary_mask < threshold] = 1.0
        binary_mask[binary_mask >= threshold] = 0.0 # cioè escludo anche la threshold stessa
        '''
        # -------------------------------
        
        '''
        plt.imshow(binary_mask, cmap='gist_gray')
        plt.title("Binary Mask")
        plt.show()
        '''
        

        '''
        # Save the binary mask

        #remove until char _
        save_filename = file.split("_")[0] + "_binary.npy"
        save_path = os.path.join(subdir, save_filename)
        np.save(save_path, binary_mask)
        '''

        # Carica l'immagine JPG
        image = Image.open(image_path).convert("RGB")  # Assicura che sia RGB
        image_np = np.array(image)  # Converti in NumPy array (HxWx3)

        # Assicurati che la maschera sia nel range corretto (0-255)
        alpha_channel = (binary_mask * 255).astype(np.uint8)

        # Converti l'immagine in RGBA aggiungendo il canale alfa
        image_rgba = np.dstack((image_np, alpha_channel))  #  Diventa HxWx4 (RGBA)

        # Crea un'immagine PIL RGBA
        image_rgba_pil = Image.fromarray(image_rgba)

        '''
        # mostra immagine tagliata
        plt.imshow(image_rgba)
        plt.title("Img")
        plt.show()
        '''

        # Salva l'immagine con trasparenza
        save_folder = save_images_threshold_folder + "_" + str(threshold)
        os.makedirs(save_folder, exist_ok=True)
        save_filename = file
        save_path = os.path.join(save_folder, save_filename)
        image_rgba_pil.save(save_path)
