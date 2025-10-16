# utils/normalizar_pose.py
import numpy as np

# Índices principais: [11, 13, 15] = braço esquerdo | [12, 14, 16] = braço direito
ARM_INDICES = {
    "left": [11, 13, 15],
    "right": [12, 14, 16]
}

def normalizar_pose(landmarks):
    """Retorna vetor 1D normalizado dos braços, ignorando cabeça e tronco."""
    all_kp = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    braços = []

    for side, idxs in ARM_INDICES.items():
        kp = all_kp[idxs]
        
        # Centraliza o braço no ombro
        kp -= kp[0]

        # Escala pelo comprimento total (ombro → punho)
        comprimento = np.linalg.norm(kp[-1] - kp[0])
        if comprimento < 1e-6:
            comprimento = 1.0
        kp /= comprimento

        braços.append(kp.flatten())

    # Retorna concatenação dos dois braços
    return np.concatenate(braços)
