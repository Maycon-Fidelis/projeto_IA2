# arquivo: coletar_dados.py

import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- CONFIGURAÇÕES ---
# AVISO: Para cada vídeo de treino que você processar, você deve:
# 1. Mudar o VIDEO_PATH.
# 2. Assegurar que o OUTPUT_CSV_PATH seja o MESMO para acumular os dados.
VIDEO_PATH = 'videos/alcancar_as_estrelas.mp4'  # <<< COLOQUE O CAMINHO DO SEU VÍDEO AQUI
OUTPUT_CSV_PATH = 'coord_videos/alcancar_as_estrelas.csv' # Caminho para o arquivo de dados acumulados
# ---------------------

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def setup_csv():
    num_coords = 33
    landmarks_header = ['class']
    for val in range(1, num_coords + 1):
        landmarks_header += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
    
    if not os.path.exists(OUTPUT_CSV_PATH):
        with open(OUTPUT_CSV_PATH, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks_header)

def save_frame_data(pose_class, results):
    try:
        landmarks = results.pose_landmarks.landmark
        
        row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten())
        row.insert(0, pose_class)
        
        with open(OUTPUT_CSV_PATH, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)
        print(f"Frame salvo para a classe: '{pose_class}'")
            
    except Exception as e:
        print(f"Nenhuma pose detectada para salvar. Erro: {e}")

setup_csv()
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {VIDEO_PATH}")
    exit()

ret, frame = cap.read()
if not ret:
    print("Não foi possível ler o primeiro frame. Verifique o arquivo de vídeo.")
    exit()

paused = True
current_class = 'down'

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Fim do vídeo. Pausando.")
                break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        status_text = "PAUSADO" if paused else "RODANDO"
        cv2.putText(image_bgr, f"Status: {status_text}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
        cv2.putText(image_bgr, f"CLASSE ATUAL: {current_class.upper()}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image_bgr, "SALVAR (s) | UP (u) | DOWN (d) | PLAY/PAUSE (espaco)", (15, image_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Coleta de Dados', image_bgr)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == 32:
            paused = not paused
        if paused:
            if key == ord('d'):
                current_class = 'down'
            if key == ord('u'):
                current_class = 'up'
            if key == ord('s'):
                save_frame_data(current_class, results)

cap.release()
cv2.destroyAllWindows()