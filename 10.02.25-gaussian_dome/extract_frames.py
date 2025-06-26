import cv2
import os

# Parametri di input
video_path = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/20250128_164309.mp4'  # Percorso del video
output_dir = '../datasets/colmap_reconstructions/brgRmSmParkFullFramesCompletePipeline/input'  # Cartella di output
fps_extract = 2  # Numero di frame da estrarre per secondo

# Crea la cartella di output, se non esiste
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Apre il video
cap = cv2.VideoCapture(video_path)

# Ottieni il frame rate del video (fps)
fps_video = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate del video: {fps_video} FPS")

# Calcola il passo (interval) per estrarre il numero giusto di frame al secondo
frame_interval = int(fps_video / fps_extract)

# Estrai i frame e salvali con il timestamp
frame_number = 0
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calcola il timestamp in secondi
    timestamp = frame_number / fps_video
    
    # Calcola ore, minuti, secondi e decimali dal timestamp
    hours = int(timestamp // 3600)  # Ore
    minutes = int((timestamp % 3600) // 60)  # Minuti
    seconds = int(timestamp % 60)  # Secondi
    milliseconds = int((timestamp * 1000) % 1000)  # Millisecondi
    
    # Crea il nome del file con il formato ore_minuti_secondi_decimali
    timestamp_str = f"{hours:02d}_{minutes:02d}_{seconds:02d}_{milliseconds:03d}"
    filename = f"{timestamp_str}.png"
    output_path = os.path.join(output_dir, filename)
    
    # Seleziona solo i frame da estrarre, in base agli FPS desiderati
    if frame_number % frame_interval == 0:
        # Salva il frame
        cv2.imwrite(output_path, frame)
        frame_count += 1
        print("salvo frame numero:", frame_number, "come", filename)
    
    frame_number += 1

# Rilascia il video
cap.release()
print(f"Estrazione completata. {frame_count} frame sono stati salvati nella cartella {output_dir}.")