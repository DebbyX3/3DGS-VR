import os

def replace_in_filenames(folder_path, old_str, new_str):
    for filename in os.listdir(folder_path):
        if old_str in filename:
            new_filename = filename.replace(old_str, new_str)
            old_file = os.path.join(folder_path, filename)
            new_file = os.path.join(folder_path, new_filename)
            os.rename(old_file, new_file)
            print(f"Renamed: {filename} → {new_filename}")

# Esempio di utilizzo
replace_in_filenames("C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\brg_rm_small_park-FullFrames\\stereo\\depth_maps",
                     "jpg",
                     "png")