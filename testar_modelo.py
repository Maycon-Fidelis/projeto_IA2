# arquivo: testar_modelo.py
import tkinter as tk
from tkinter import ttk, font
import cv2
import mediapipe as mp
import pickle
import numpy as np
from PIL import Image, ImageTk
import os
import random

# --- CARREGAMENTO DO MODELO E MEDIAPIPE ---
try:
    with open("modelos/alcancar_as_estrelas.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("ERRO: Arquivo 'pose_classifier_model.pkl' não encontrado.")
    exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseApp(tk.Tk):
    """Classe principal que gerencia a aplicação e as telas."""
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
        """Mostra uma tela, parando a lógica da tela anterior."""
        frame = self.frames[cont]
        if self.previous_frame and hasattr(self.previous_frame, 'stop_mission'):
            self.previous_frame.stop_mission()
        
        if hasattr(frame, 'start_mission'):
            frame.start_mission()
        if hasattr(frame, 'set_results') and data:
            frame.set_results(data)

        frame.tkraise()
        self.previous_frame = frame

# (LevelSelectionFrame permanece o mesmo)
class LevelSelectionFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg='#ecf0f1')
        label = tk.Label(self, text="Selecione a Missão", font=('Nunito', 24, 'bold'), bg='#ecf0f1', fg='#3498db')
        label.pack(pady=40, padx=10)
        mission1_btn = tk.Button(self, text="Fase 1: Levantamento", font=('Nunito', 14, 'bold'),
                                 bg='#f1c40f', fg='#2c3e50',
                                 command=lambda: controller.show_frame(MissionFrame))
        mission1_btn.pack(pady=20, padx=20, ipadx=10, ipady=10)

class MissionFrame(tk.Frame):
    """Tela principal da missão com gamificação aprimorada."""
    MISSION_GOAL = 10
    STAR_THRESHOLDS = (3, 6, 10)

    def __init__(self, parent, controller):
        super().__init__(parent, bg='#2c3e50')
        self.controller = controller
        
        self.idle_frames = self.load_animation_frames('assets/idle')
        self.action_frames = self.load_animation_frames('assets/action')
        self.star_empty_img = self.load_ui_image('assets/ui/star_empty.png', (64, 64))
        self.star_filled_img = self.load_ui_image('assets/ui/star_filled.png', (64, 64))
        
        self.is_mission_running = False
        self.pose_processor = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)

        main_panel = tk.Frame(self, bg='#2c3e50')
        main_panel.pack(fill="both", expand=True, padx=20, pady=20)
        
        video_panel = tk.Frame(main_panel, bg='#2c3e50')
        video_panel.pack(side="left", fill="both", expand=True)
        self.video_label = tk.Label(video_panel, bg='black')
        self.video_label.pack(padx=10, pady=10, fill="both", expand=True)
        
        game_panel = tk.Frame(main_panel, bg='#34495e', width=400)
        game_panel.pack(side="right", fill="y", padx=10)
        game_panel.pack_propagate(False)

        # --- NOVOS WIDGETS DE GAMIFICAÇÃO ---
        self.goal_label = tk.Label(game_panel, text=f"Meta: 0 / {self.MISSION_GOAL}", font=('Nunito', 22, 'bold'), bg='#34495e', fg='white')
        self.goal_label.pack(pady=(20, 10))
        
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("green.Horizontal.TProgressbar", foreground='#2ecc71', background='#2ecc71', thickness=25)
        self.progress_bar = ttk.Progressbar(game_panel, orient="horizontal", length=300, mode="determinate", maximum=self.MISSION_GOAL, style="green.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=10)
        
        # <<< NOVO LABEL PARA FEEDBACK DE TEXTO >>>
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
        
    def load_ui_image(self, path, size):
        try:
            return ImageTk.PhotoImage(Image.open(path).resize(size, Image.Resampling.LANCZOS))
        except: return None
        
    def load_animation_frames(self, path):
        frames = []
        try:
            files = sorted(os.listdir(path), key=lambda x: int(''.join(filter(str.isdigit, x))))
            for filename in files:
                img = Image.open(os.path.join(path, filename)).resize((300, 300), Image.Resampling.LANCZOS)
                frames.append(ImageTk.PhotoImage(img))
        except: print(f"AVISO: Pasta de assets não encontrada em '{path}'.")
        return frames

    def start_mission(self):
        if self.is_mission_running: return
        self.is_mission_running = True
        self.counter = 0
        self.stage = 'down'
        self.cap = cv2.VideoCapture(0)
        self.reset_ui()
        self.update_frame()
        self.animate_hero()

    def stop_mission(self):
        if not self.is_mission_running: return
        self.is_mission_running = False
        if self.cap: self.cap.release()
        self.cap = None

    def reset_ui(self):
        """Reseta a interface para o início da missão."""
        self.goal_label.config(text=f"Meta: 0 / {self.MISSION_GOAL}")
        self.progress_bar['value'] = 0
        self.update_feedback_text("Prepare-se para começar a missão!")
        for star_label in self.star_labels:
            star_label.config(image=self.star_empty_img)
        self.speech_bubble.pack_forget()

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
            pose_class = model.predict(X)[0]

            if pose_class == 'up' and self.stage == 'down':
                self.stage = "up"
                self.counter += 1
                self.on_rep_success()
            elif pose_class == 'down' and self.stage == 'up':
                self.stage = "down"
                self.update_feedback_text("Ótimo! Agora desça o peso controladamente.")


            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
        self.video_label.config(image=img_tk)
        self.video_label.image = img_tk
        self.after(10, self.update_frame)

    def on_rep_success(self):
        """Atualiza todos os elementos de gamificação após uma repetição."""
        self.goal_label.config(text=f"Meta: {self.counter} / {self.MISSION_GOAL}")
        
        # <<< LÓGICA DA BARRA DE PROGRESSO ESTRATÉGICA >>>
        if self.counter < self.MISSION_GOAL:
            # Para cada repetição antes da final, preenche a barra "quase" até o próximo nível
            progress_value = self.counter - 0.1
            self.update_feedback_text(random.choice(["Continue assim!", "Mantenha o ritmo!", "Falta pouco!"]))
        else:
            # Na repetição final, a barra se completa!
            progress_value = self.MISSION_GOAL
            self.update_feedback_text("Você conseguiu!")
        self.progress_bar['value'] = progress_value
        
        self.update_stars()
        self.trigger_hero_action()

        if self.counter >= self.MISSION_GOAL:
            self.mission_complete()

    def update_stars(self):
        stars_earned = 0
        for i, threshold in enumerate(self.STAR_THRESHOLDS):
            if self.counter >= threshold:
                self.star_labels[i].config(image=self.star_filled_img)
                stars_earned += 1
        return stars_earned
        
    def update_feedback_text(self, text):
        """Atualiza o label central de feedback."""
        self.feedback_label.config(text=text)

    def mission_complete(self):
        stars = self.update_stars()
        # Atraso de 1 segundo antes de mudar de tela para o jogador ver a barra cheia
        self.after(1000, lambda: self.controller.show_frame(MissionCompleteFrame, data={'stars': stars}))

    def animate_hero(self):
        if not self.is_mission_running: return
        frames_to_play = self.idle_frames
        if getattr(self, 'animation_state', 'idle') == 'action':
            frames_to_play = self.action_frames
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
            if getattr(self, 'animation_state', 'idle') == 'action':
                self.animation_state = 'idle'
        
        # <<< ANIMAÇÃO MAIS LENTA >>>
        self.after(150, self.animate_hero)
        
    def trigger_hero_action(self):
        self.animation_state = 'action'
        self.current_frame_index = 0
        motivational_phrases = ["Graças a você, levantei a rocha!", "Sua força me deu energia!", "Juntos somos imbatíveis!", "Ótimo trabalho! Senti nosso poder!"]
        self.speech_bubble.config(text=random.choice(motivational_phrases))
        self.speech_bubble.pack(pady=20, before=self.hero_label)
        
        # <<< FALA FICA MAIS TEMPO NA TELA >>>
        self.after(3500, lambda: self.speech_bubble.pack_forget())

class MissionCompleteFrame(tk.Frame):
    """Tela de vitória exibida ao final da missão."""
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
            
        tk.Button(self, text="Voltar ao Menu", font=('Nunito', 16, 'bold'),
                  command=lambda: controller.show_frame(LevelSelectionFrame)).pack(pady=50)

    def load_ui_image(self, path, size):
        try: return ImageTk.PhotoImage(Image.open(path).resize(size, Image.Resampling.LANCZOS))
        except: return None

    def set_results(self, data):
        """Atualiza as estrelas com base no resultado da missão."""
        stars_earned = data.get('stars', 0)
        for i in range(3):
            if i < stars_earned:
                self.result_star_labels[i].config(image=self.star_filled_img)
            else:
                self.result_star_labels[i].config(image=self.star_empty_img)

if __name__ == "__main__":
    app = PoseApp()
    app.mainloop()