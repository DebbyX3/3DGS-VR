import os
from PIL import Image

def convert_jpg_to_png(input_folder, output_folder=None):
    if output_folder is None:
        output_folder = input_folder  # salva i PNG nella stessa cartella

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            jpg_path = os.path.join(input_folder, filename)
            img = Image.open(jpg_path).convert("RGB")  # garantisce che non ci siano problemi con modalità
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(output_folder, png_filename)
            img.save(png_path)
            print(f"Converted {filename} → {png_filename}")

# Esempio di utilizzo
convert_jpg_to_png("C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\fields\\imagesJPGOld", 
                   output_folder="C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\fields\\input")