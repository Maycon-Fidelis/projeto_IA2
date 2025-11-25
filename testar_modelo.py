import tkinter as tk
from tkinter import ttk, font
import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image, ImageTk
import os
import random
import pygame 

# --- CONFIGURAÇÃO CENTRAL DE EXERCÍCIOS ---
EXERCISES = {
    "estrelas": {
        "name": "Alcançar as Estrelas",
        "model_path": "modelos/alcancar_as_estrelas.pkl",
        "logic": ['down', 'up']
    },
    "asas": {
        "name": "Asas de Super-Herói",
        "model_path": "modelos/asas_de_super_heroi.pkl",
        "logic": ['middle', 'up', 'middle']
    },
    "parede": {
        "name": "Empurrar Parede",
        "model_path": "modelos/empurrar_parede.pkl",
        "logic": ['down', 'push']
    }
}
# -------------------------------------------

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURAÇÃO DE ÁUDIO ---
# Define o diretório base do script para carregamento seguro
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pygame.mixer_available = False
SOUNDS = {}

try:
    pygame.mixer.init()
    pygame.mixer_available = True
except pygame.error as e:
    print(f"ERRO ao inicializar o mixer de áudio: {e}")

if pygame.mixer_available:
    try:
        # Carrega os arquivos .mp3 usando o caminho absoluto
        SOUNDS = {
            'success': pygame.mixer.Sound(os.path.join(BASE_DIR, 'assets', 'audio', 'success.mp3')),  
            'transition': pygame.mixer.Sound(os.path.join(BASE_DIR, 'assets', 'audio', 'transition.mp3')), 
            'start': pygame.mixer.Sound(os.path.join(BASE_DIR, 'assets', 'audio', 'start.mp3')), 
            'complete': pygame.mixer.Sound(os.path.join(BASE_DIR, 'assets', 'audio', 'complete.mp3'))
        }
    except pygame.error as e:
        print(f"AVISO: Falha ao carregar um dos arquivos de áudio. Verifique se estão na pasta assets/audio/: {e}")
        SOUNDS = {}
    
def play_sound(sound_key):
    """Toca o som se o mixer estiver disponível e o som tiver sido carregado."""
    if pygame.mixer_available and sound_key in SOUNDS:
        SOUNDS[sound_key].play()
# -----------------------------

class PoseApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Missões do Herói IA")
        self.geometry("1280x800")
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        self.previous_frame = None
        for F in (LevelSelectionFrame, MissionFrame, MissionCompleteFrame):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(LevelSelectionFrame)

    def show_frame(self, cont, data=None):
        frame = self.frames[cont]
        if self.previous_frame and hasattr(self.previous_frame, 'stop_mission'):
            self.previous_frame.stop_mission()
        if cont == MissionFrame and data:
            frame.configure_mission(data)
            frame.start_mission()
        if cont == MissionCompleteFrame and data:
            play_sound('complete')
            frame.set_results(data)
        frame.tkraise()
        self.previous_frame = frame

class LevelSelectionFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg='#ecf0f1')
        label = tk.Label(self, text="Selecione a Missão", font=('Nunito', 24, 'bold'), bg='#ecf0f1', fg='#3498db')
        label.pack(pady=40, padx=10)
        button_font = font.Font(family='Nunito', size=14, weight='bold')
        for key, exercise_data in EXERCISES.items():
            btn = tk.Button(self, text=exercise_data['name'],
                            font=button_font, bg='#f1c40f', fg='#2c3e50',
                            command=lambda data=exercise_data: controller.show_frame(MissionFrame, data=data))
            btn.pack(pady=15, padx=20, ipadx=10, ipady=10)

class MissionFrame(tk.Frame):
    MISSION_GOAL = 5
    STAR_THRESHOLDS = (1, 3, 5)

    def __init__(self, parent, controller):
        super().__init__(parent, bg='#2c3e50')
        self.controller = controller
        
        self.model = None
        self.exercise_logic = []
        self.logic_index = 0 
        self.stage = ""
        
        self.idle_frames = self.load_animation_frames('assets/idle')
        self.action_frames = self.load_animation_frames('assets/action')
        self.star_empty_img = self.load_ui_image('assets/ui/star_empty.png', (64, 64))
        self.star_filled_img = self.load_ui_image('assets/ui/star_filled.png', (64, 64))
        
        self.is_mission_running = False
        # Modelo de complexidade 0 é o mais leve e rápido
        self.pose_processor = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4, model_complexity=0)

        # --- Layout ---
        main_panel = tk.Frame(self, bg='#2c3e50')
        main_panel.pack(fill="both", expand=True, padx=20, pady=20)
        video_panel = tk.Frame(main_panel, bg='#2c3e50')
        video_panel.pack(side="left", fill="both", expand=True)
        self.video_label = tk.Label(video_panel, bg='black')
        self.video_label.pack(padx=10, pady=10, fill="both", expand=True)
        game_panel = tk.Frame(main_panel, bg='#34495e', width=400)
        game_panel.pack(side="right", fill="y", padx=10)
        game_panel.pack_propagate(False)
        
        self.goal_label = tk.Label(game_panel, text=f"Meta: 0 / {self.MISSION_GOAL}", font=('Nunito', 22, 'bold'), bg='#34495e', fg='white')
        self.goal_label.pack(pady=(20, 10))
        
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("green.Horizontal.TProgressbar", foreground='#2ecc71', background='#2ecc71', thickness=25)
        self.progress_bar = ttk.Progressbar(game_panel, orient="horizontal", length=300, mode="determinate", maximum=self.MISSION_GOAL, style="green.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=10)
        
        # <<< NOVO: Label da Pontuação de Confiança >>>
        self.performance_label = tk.Label(game_panel, text="Confiança: 0%", 
                                          font=('Nunito', 16, 'bold'), 
                                          bg='#34495e', 
                                          fg='#f1c40f') 
        self.performance_label.pack(pady=10)
        
        self.feedback_label = tk.Label(game_panel, text="", font=('Nunito', 14, 'italic'), bg='#34495e', fg='white', wraplength=350)
        self.feedback_label.pack(pady=15)
        
        star_frame = tk.Frame(game_panel, bg='#34495e')
        star_frame.pack(pady=20)
        self.star_labels = []
        for _ in range(3):
            star_label = tk.Label(star_frame, image=self.star_empty_img, bg='#34495e')
            star_label.pack(side='left', padx=5)
            self.star_labels.append(star_label)
            
        self.hero_label = tk.Label(game_panel, bg='#34495e')
        self.hero_label.pack(expand=True)
        self.speech_bubble = tk.Label(game_panel, text="", bg='white', fg='black', font=('Nunito', 12, 'bold'), wraplength=300)

    def configure_mission(self, exercise_data):
        try:
            with open(exercise_data['model_path'], "rb") as f:
                self.model = pickle.load(f)
            self.exercise_logic = exercise_data['logic']
        except Exception as e:
            self.model = None
            self.exercise_logic = []
            print(f"ERRO ao carregar o modelo: {e}")
            self.feedback_label.config(text=f"Erro ao carregar o modelo!")

    def start_mission(self):
        if self.is_mission_running or self.model is None: return
        self.is_mission_running = True
        self.counter = 0
        self.logic_index = 0 
        self.stage = "" 
        self.cap = cv2.VideoCapture(0)
        self.reset_ui()
        play_sound('start') 
        self.update_frame()
        self.animate_hero()

    def update_frame(self):
        if not self.is_mission_running: return
        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_frame)
            return
            
        frame = cv2.resize(frame, (640, 480))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_processor.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            X = np.array(row).reshape(1, -1)
            
            # --- Predição com Probabilidade e Confiança ---
            probabilities = self.model.predict_proba(X)[0]
            pose_class_index = np.argmax(probabilities)
            pose_class = self.model.classes_[pose_class_index]
            confidence_score = probabilities[pose_class_index]
            # ----------------------------------------------

            # Exibe a pontuação de confiança na tela
            score_percent = int(confidence_score * 100)
            self.performance_label.config(text=f"Confiança: {score_percent}%")

            # --- Lógica de Contagem e Feedback Acurado ---
            expected_stage = self.exercise_logic[self.logic_index]
            
            MIN_CONFIDENCE = 0.8 # Limite de confiança para aceitar a pose
            
            if pose_class == expected_stage and confidence_score > MIN_CONFIDENCE:
                # Pose correta e com boa confiança: avança.
                self.stage = pose_class
                self.logic_index += 1
                
                next_stage_name = self.exercise_logic[self.logic_index % len(self.exercise_logic)]
                self.update_feedback_text(f"Correto! Próximo passo: **{next_stage_name.upper()}**")
                
                play_sound('transition')
                
                # Repetição Completa
                if self.logic_index >= len(self.exercise_logic):
                    self.logic_index = 0
                    self.counter += 1
                    self.on_rep_success()
            
            elif pose_class != expected_stage and confidence_score > MIN_CONFIDENCE:
                # Pose errada, mas o modelo tem certeza: feedback de correção.
                self.update_feedback_text(f"Mantenha a postura! O modelo espera **{expected_stage.upper()}** agora.")
            
            elif confidence_score <= MIN_CONFIDENCE:
                 # Confiança baixa: feedback para melhorar a visibilidade/postura.
                self.update_feedback_text("Ajuste a posição! Confiança baixa. Certifique-se de que todas as partes do corpo estejam visíveis.")
            # ------------------------------------------------------------------

            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk
        self.after(10, self.update_frame)
        
    def load_ui_image(self, path, size):
        try: return ImageTk.PhotoImage(Image.open(path).resize(size, Image.Resampling.LANCZOS))
        except: return None
        
    def load_animation_frames(self, path):
        frames = []
        try:
            # Usa o caminho absoluto para carregar assets
            full_path = os.path.join(BASE_DIR, path)
            files = sorted(os.listdir(full_path), key=lambda x: int(''.join(filter(str.isdigit, x))))
            for filename in files:
                img = Image.open(os.path.join(full_path, filename)).resize((300, 300), Image.Resampling.LANCZOS)
                frames.append(ImageTk.PhotoImage(img))
        except: print(f"AVISO: Pasta de assets não encontrada em '{path}'.")
        return frames
        
    def stop_mission(self):
        if not self.is_mission_running: return
        self.is_mission_running = False
        if self.cap: self.cap.release()
        self.cap = None
        
    def reset_ui(self):
        self.goal_label.config(text=f"Meta: 0 / {self.MISSION_GOAL}")
        self.progress_bar['value'] = 0
        self.performance_label.config(text="Confiança: 0%") # Reset da Confiança
        self.update_feedback_text("Prepare-se para começar a missão!")
        for star_label in self.star_labels: star_label.config(image=self.star_empty_img)
        self.speech_bubble.pack_forget()
        
    def on_rep_success(self):
        self.goal_label.config(text=f"Meta: {self.counter} / {self.MISSION_GOAL}")
        
        if self.counter < self.MISSION_GOAL:
            progress_value = self.counter - 0.1
            self.update_feedback_text("Repetição Completa! Ótimo!")
            play_sound('success')
        else:
            progress_value = self.MISSION_GOAL
            self.update_feedback_text("Você conseguiu!")
            
        self.progress_bar['value'] = progress_value
        self.update_stars()
        self.trigger_hero_action()
        
        if self.counter >= self.MISSION_GOAL: self.mission_complete()
        
    def update_stars(self):
        stars_earned = 0
        for i, threshold in enumerate(self.STAR_THRESHOLDS):
            if self.counter >= threshold:
                self.star_labels[i].config(image=self.star_filled_img)
                stars_earned += 1
        return stars_earned
        
    def update_feedback_text(self, text):
        self.feedback_label.config(text=text)
        
    def mission_complete(self):
        stars = self.update_stars()
        self.after(1000, lambda: self.controller.show_frame(MissionCompleteFrame, data={'stars': stars}))
        
    def animate_hero(self):
        if not self.is_mission_running: return
        frames_to_play = self.idle_frames
        if getattr(self, 'animation_state', 'idle') == 'action': frames_to_play = self.action_frames
        
        if not frames_to_play:
            self.after(150, self.animate_hero)
            return
            
        self.current_frame_index = getattr(self, 'current_frame_index', 0)
        frame_image = frames_to_play[self.current_frame_index]
        self.hero_label.config(image=frame_image)
        self.hero_label.image = frame_image
        self.current_frame_index += 1
        
        if self.current_frame_index >= len(frames_to_play):
            self.current_frame_index = 0
            if getattr(self, 'animation_state', 'idle') == 'action': self.animation_state = 'idle'
            
        self.after(150, self.animate_hero)
        
    def trigger_hero_action(self):
        self.animation_state = 'action'
        self.current_frame_index = 0
        motivational_phrases = ["Graças a você, ajudei a fortalecer o mundo!", "Sua energia é contagiante!", "Juntos somos uma super-equipe!", "Ótimo movimento! Senti o poder!"]
        self.speech_bubble.config(text=random.choice(motivational_phrases))
        self.speech_bubble.pack(pady=20, before=self.hero_label)
        self.after(3500, lambda: self.speech_bubble.pack_forget())

class MissionCompleteFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg='#2ecc71')
        self.controller = controller
        self.star_empty_img = self.load_ui_image('assets/ui/star_empty.png', (100, 100))
        self.star_filled_img = self.load_ui_image('assets/ui/star_filled.png', (100, 100))
        tk.Label(self, text="MISSÃO CONCLUÍDA!", font=('Nunito', 40, 'bold'), bg='#2ecc71', fg='white').pack(pady=(80, 20))
        self.star_frame = tk.Frame(self, bg='#2ecc71')
        self.star_frame.pack(pady=40)
        self.result_star_labels = []
        for _ in range(3):
            star_label = tk.Label(self.star_frame, image=self.star_empty_img, bg='#2ecc71')
            star_label.pack(side='left', padx=10)
            self.result_star_labels.append(star_label)
        tk.Button(self, text="Voltar ao Menu", font=('Nunito', 16, 'bold'), command=lambda: controller.show_frame(LevelSelectionFrame)).pack(pady=50)
        
    def load_ui_image(self, path, size):
        try: return ImageTk.PhotoImage(Image.open(os.path.join(BASE_DIR, path)).resize(size, Image.Resampling.LANCZOS))
        except: return None
        
    def set_results(self, data):
        stars_earned = data.get('stars', 0)
        for i in range(3):
            if i < stars_earned: self.result_star_labels[i].config(image=self.star_filled_img)
            else: self.result_star_labels[i].config(image=self.star_empty_img)

if __name__ == "__main__":
    app = PoseApp()
    app.mainloop()