import os

def trova_file_mancanti(cartella1, cartella2):
    # Elenco dei file in entrambe le cartelle (solo nomi, non path completi)
    files1 = set(os.listdir(cartella1))
    files2 = set(os.listdir(cartella2))

    # Trova i file presenti in cartella1 ma non in cartella2
    mancanti = files1 - files2

    return sorted(mancanti)

# Esempio di utilizzo
cartella1 = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\fields\\input"
cartella2 = "C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\fields\\images_threshold_10"

file_mancanti = trova_file_mancanti(cartella1, cartella2)

print("File presenti in cartella1 ma non in cartella2:")
for f in file_mancanti:
    print(f)
