# Nome del file da modificare
file_path = 'C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\fields\\sparse\\0\\images_old.txt'

# Leggi il contenuto
with open(file_path, 'r') as f:
    content = f.read()

# Sostituisci tutte le occorrenze di ".jpg" con ".png"
new_content = content.replace('.jpg', '.png')

new_file_path = 'C:\\Users\\User\\Desktop\\Gaussian Splatting\\3DGS-VR\\datasets\\colmap_reconstructions\\fields\\sparse\\0\\images.txt'

with open(new_file_path, 'w') as f:
    f.write(new_content)

print("Sostituzione completata!")
