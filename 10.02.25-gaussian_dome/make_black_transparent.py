from PIL import Image
import os

def rendi_nero_trasparente(cartella_input, cartella_output):
    for filename in os.listdir(cartella_input):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            continue

        percorso = os.path.join(cartella_input, filename)
        img = Image.open(percorso).convert("RGBA")

        dati = img.getdata()
        nuovi_dati = []
        for pixel in dati:
            r, g, b, a = pixel
            if r == 0 and g == 0 and b == 0:
                nuovi_dati.append((0, 0, 0, 0))  # completamente trasparente
            else:
                nuovi_dati.append((r, g, b, a))

        img.putdata(nuovi_dati)

        # Salva con lo stesso nome ma estensione .png
        nome_senza_estensione = os.path.splitext(filename)[0]
        nuovo_path = os.path.join(cartella_output, nome_senza_estensione + ".png")
        img.save(nuovo_path, "PNG")
        print(f"Salvata immagine con trasparenza: {nuovo_path}")

# Esempio d'uso:
cartella_input = "../datasets/colmap_reconstructions/ice_lab_360_2_500ImgsCirca_transp_AddBackLab/undistorted"
cartella_output = "../datasets/colmap_reconstructions/ice_lab_360_2_500ImgsCirca_transp_AddBackLab/imagesTransparent"
rendi_nero_trasparente(cartella_input, cartella_output)