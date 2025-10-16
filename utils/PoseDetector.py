import cv2
import mediapipe as mp
import numpy as np

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth=True,
                 detection_con=0.5, track_con=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=mode,
                                      model_complexity=complexity,
                                      smooth_landmarks=smooth,
                                      min_detection_confidence=detection_con,
                                      min_tracking_confidence=track_con)
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks,
                                        self.mp_pose.POSE_CONNECTIONS)
        return img

    def get_landmark_positions(self, img):
        self.landmark_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                px, py = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append([id, px, py])
        return self.landmark_list

    def calculate_angle(self, p1_idx, p2_idx, p3_idx):
        if len(self.landmark_list) == 0:
            return None
        
        # As coordenadas agora estão em landmark_list[indice][1] e [2]
        _, x1, y1 = self.landmark_list[p1_idx]
        _, x2, y2 = self.landmark_list[p2_idx]
        _, x3, y3 = self.landmark_list[p3_idx]

        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = np.abs(np.degrees(radians))
        if angle > 180:
            angle = 360 - angle
        return angle