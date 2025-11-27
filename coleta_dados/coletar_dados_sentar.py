import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# --- CONFIGURAÇÕES ADAPTADAS PARA SENTAR-E-LEVANTAR ---
# 1. Altere o caminho do vídeo para o seu arquivo de sentar/levantar
VIDEO_PATH = './videos/levantar_abaixar.mp4' 
# 2. Altere o caminho do CSV de saída
OUTPUT_CSV_PATH = './coord_videos/sentar_e_levantar.csv' 
# -----------------------------------------------------

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# As funções setup_csv e save_frame_data não precisam de alterações.
# Elas continuam a funcionar perfeitamente com qualquer nome de classe.

def setup_csv():
    """Prepara o arquivo CSV com o cabeçalho correto, se ele não existir."""
    
    # Cria o diretório se não existir
    output_dir = os.path.dirname(OUTPUT_CSV_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    num_coords = 33
    landmarks_header = ['class']
    for val in range(1, num_coords + 1):
        landmarks_header += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
    
    if not os.path.exists(OUTPUT_CSV_PATH):
        with open(OUTPUT_CSV_PATH, mode='w', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks_header)

def save_frame_data(pose_class, results):
    """Salva os landmarks do frame atual no arquivo CSV com a classe fornecida."""
    try:
        landmarks = results.pose_landmarks.landmark
        
        row = list(np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten())
        row.insert(0, pose_class)
        
        with open(OUTPUT_CSV_PATH, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)
        print(f"Frame salvo para a classe: '{pose_class}'")
            
    except Exception as e:
        print(f"Nenhuma pose detectada para salvar.")

# --- INÍCIO DA LÓGICA PRINCIPAL ---
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
# Altere a classe inicial:
current_class = 'sentado' # Estado Sentado (equivalente ao 'down')

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
        # Exibe a classe atual:
        cv2.putText(image_bgr, f"CLASSE ATUAL: {current_class.upper()}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Atualiza o texto de ajuda:
        help_text = "SALVAR (s) | EM_PE (u) | TRANSICAO (m) | SENTADO (d) | PLAY/PAUSE (espaco)"
        cv2.putText(image_bgr, help_text, (15, image_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Coleta de Dados Interativa', image_bgr)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == 32:
            paused = not paused
        
        if paused:
            # Altera o nome da classe salva ao pressionar a tecla
            if key == ord('d'):
                current_class = 'sentado'
                print("Classe alterada para 'sentado'")
            if key == ord('m'):
                current_class = 'transicao'
                print("Classe alterada para 'transicao'")
            if key == ord('u'):
                current_class = 'em_pe'
                print("Classe alterada para 'em_pe'")
            if key == ord('s'):
                save_frame_data(current_class, results)

cap.release()
cv2.destroyAllWindows()