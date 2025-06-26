import os
import shutil
import random

def sposta_immagini_random(sorgente, destinazione, percentuale=0.5, estensioni_immagini=None):
    if estensioni_immagini is None:
        estensioni_immagini = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']

    # Crea la cartella di destinazione se non esiste
    os.makedirs(destinazione, exist_ok=True)

    # Lista tutte le immagini nella cartella sorgente
    immagini = [f for f in os.listdir(sorgente)
                if os.path.isfile(os.path.join(sorgente, f)) and
                os.path.splitext(f)[1].lower() in estensioni_immagini]

    # Calcola il numero di immagini da spostare
    n_da_spostare = max(1, int(len(immagini) * percentuale))

    # Seleziona immagini a caso
    immagini_scelte = random.sample(immagini, n_da_spostare)

    # Sposta le immagini selezionate
    for img in immagini_scelte:
        src_path = os.path.join(sorgente, img)
        dst_path = os.path.join(destinazione, img)
        shutil.move(src_path, dst_path)
        print(f"Spostata: {img}")

    print(f"{len(immagini_scelte)} immagini spostate su {len(immagini)} totali.")

# Esempio d'uso:
sorgente = "../datasets/colmap_reconstructions/ice_lab_360_2_500ImgsCirca_transp_AddBackLab/images/new to remove bg"
destinazione = "../datasets/colmap_reconstructions/ice_lab_360_2_500ImgsCirca_transp_AddBackLab/images/movedBecauseTooMany"
sposta_immagini_random(sorgente, destinazione)
